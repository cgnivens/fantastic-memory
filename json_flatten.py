import re

from collections import namedtuple
from functools import reduce
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType,
    ArrayType,
    StringType,
)
from pyspark.sql.dataframe import DataFrame
from typing import (
    Optional,
    Iterator,
    Tuple,
    List,
    Union,
)
# can be s3 or s3a for local testing
S3_PREFIX = 's3'

DFColPair = namedtuple("DFColPair", ["df", "column"])


def get_json_schema(
    df: DataFrame,
    *columns: str,
    multiline: bool=True
) -> List[StructType]:
    """Return schema of a json-string formatted set of columns

    Takes an arbitrary number of columns to apply this to
    """
    return [(
        spark
        .read
        .json(
            df.rdd.map(lambda row: row[c]), multiLine=multiline
        ).schema
    ) for c in columns]


def create_id_col(
    df: DataFrame,
    name: str,
    id_col: str,
    exclude_datetimes: bool=True,
    exclude: Optional[List[str]]=None
) -> DataFrame:
    """Creates an id using a sha2 hash of column 'name'"""
    exclude = set(exclude) if exclude is not None else set()

    if exclude_datetimes:
        # covers created_at, cancelled_at, updated_at, discarded_at
        exclude |= {c for c in df.columns if c.endswith('ed_at')}

    if id_col in df.columns:
        newdf = df
    elif name not in df.columns:
        cols = (F.col(c).cast(StringType()) for c in df.columns if c not in exclude)
        newdf = (
            df
            .withColumn(id_col, F.sha2(F.concat_ws('_', *cols), 256))
        )
    else:
        newdf = (
            df
            .withColumn(id_col, F.sha2(F.col(name).cast(StringType()), 256))
        )

    return newdf.filter(F.col(id_col).isNotNull())


def flattener(
    df: DataFrame,
    df_name: Optional[str]=None,
    group_id: Optional[str]=None,
    cascade_col: Optional[Union[str, List[str]]]=None,
) -> Iterator[Tuple[str, DataFrame]]:
    """Flattens a deeply-nested schema into multiple single-level dataframes with corresponding IDs

    ArrayTypes will nest into corresponding group_ids which will allow for a
    one-to-many relationship to join back together.

    This returns a generator that can be consumed by `dict()`

    Parameters
    ----------
    df: DataFrame -> Dataframe to be flattened
    df_name: Optional str -> name of the dataframe to be returned
        as the key in the dict
    group_id: Optional str -> Optional group_id for mapping to array/struct
        types
    cascade_col: Optional[str or list] -> single column or list of columns
        to cascade down to flattened dataframes
    """
    if cascade_col is None:
        cascade = []
    elif isinstance(cascade_col, str):
        cascade = [cascade_col]
    elif isinstance(cascade_col, list):
        cascade = cascade_col
    else:
        raise TypeError(
            f"Expected either str or list for cascade_col, got {type(cascade_col)}"
        )

    for name, type_ in map(lambda f: (f.name, f.dataType), df.schema.fields):
        id_col = f'{name}_id'

        if isinstance(type_, StructType):
            if group_id in df.columns and group_id != id_col:
                cols = (f'{name}.*', id_col, group_id, *cascade)
            else:
                cols = (f'{name}.*', id_col, *cascade)

            if id_col not in df.columns:
                df = create_id_col(df, name, id_col)

            yield from flattener(df.select(*cols), name, group_id=id_col, cascade_col=cascade_col)

            df = df.drop(name)
        elif isinstance(type_, ArrayType):

            if isinstance(type_.elementType, StructType):
                newdf = (
                    df
                    .select(F.explode(name).alias('exploded'), group_id, *cascade)
                    .select('exploded.*', group_id, *cascade)
                )
            else:
                newdf = create_id_col(df, name, id_col)
                newdf = newdf.select(F.explode(name), group_id, id_col, *cascade)

            yield from flattener(newdf, name, group_id=group_id, cascade_col=cascade_col)

            df = df.drop(name)
        else:
            continue

    yield (df_name, df)



def write_tables(
    collection: Iterator[Tuple[str, DFColPair]],
    bucket: str,
    prefix: str
):
    """Writes dataframes to S3. Takes an iterable of name: dataframe pairs

    Prefix is the start of the key denoting common 'folder' structures for shared items
    Partition marks if there is a group_id to be considered when writing elements to
      S3. This can speed up writing and reading
    """
    for table_name, (df, partition_col) in collection:
        df = df.select([F.col(c).alias(strip_chars(c)) for c in df.columns])

        table_name = strip_chars(table_name)

        table_endpoint = f'{S3_PREFIX}://{bucket}/{prefix.strip("/")}/{table_name}'

        if partition_col:
            writer = (
                df
                .repartition(partition_col)
                .write
                .format('parquet')
                .partitionBy(partition_col)
                .mode("overwrite")
                .option("partitionOverwriteMode", "dynamic")
            )
        else:
            writer = (
                df
                .write
                .format('parquet')
                .mode('overwrite')
            )

        writer.save(table_endpoint)



def strip_chars(c: str) -> str:
    """Remove erroneous characters from strings, also convert space to _"""
    chars = f'{chr(8211)}()[]\'",.!?\\`'

    c = (
        c
        .lower()
        .strip()
        .replace(' ', '_')
        .replace('-', '_')
        .replace('/', '_')
    )

    val = reduce(
        lambda x, y: x.replace(y, ''),
        chars,
        c
    )

    return re.sub('_{2,}', '_', val)
