from  pyspark.sql import SQLContext
from pyspark import SparkContext

sc = SparkContext()
sqlContext = SQLContext(sc)

df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('student_marks.csv')
df.registerTempTable("students")
results = sqlContext.sql("select sum(English) from students group by Gender")
print(results.collect())