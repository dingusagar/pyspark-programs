from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix, BlockMatrix
from  pyspark.sql import SQLContext
from pyspark import SparkContext


sc = SparkContext()
rows = sc.parallelize([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).zipWithIndex()

# need a SQLContext() to generate an IndexedRowMatrix from RDD
sqlContext = SQLContext(sc)
rows = IndexedRowMatrix( \
    rows \
    .map(lambda row: IndexedRow(row[1], row[0])) \
    ).toBlockMatrix()

mat_product = rows.multiply(rows)
result = mat_product.toLocalMatrix()
print("Matrix Product \n",result)
mat_sum = rows.add(rows)
result = mat_sum.toLocalMatrix()
print("Matrix Sum \n",result)
