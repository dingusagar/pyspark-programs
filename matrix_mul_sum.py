from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix, BlockMatrix
from  pyspark.sql import SQLContext
from pyspark import SparkContext


sc = SparkContext()
rows = sc.parallelize([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).zipWithIndex()

# need a SQLContext() to generate an IndexedRowMatrix from RDD
sqlContext = SQLContext(sc)
block_matrix = IndexedRowMatrix( \
    rows \
    .map(lambda row: IndexedRow(row[1], row[0])) \
    ).toBlockMatrix()

mat_product = block_matrix.multiply(block_matrix)
result = mat_product.toLocalMatrix()
print("Matrix Product \n",result)
mat_sum = block_matrix.add(block_matrix)
result = mat_sum.toLocalMatrix()
print("Matrix Sum \n",result)

mat_transpose = block_matrix.transpose()
result = mat_transpose.toLocalMatrix()
print("Matrix Transpose \n",result)