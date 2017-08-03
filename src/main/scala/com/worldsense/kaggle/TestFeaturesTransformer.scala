package com.worldsense.kaggle

/** Read the data files from http://kaggle.com/c/quora-question-pairs.
  * The contents of both train and test files are loaded in a dataframe with minimum data transformations.
  */

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.{lit, udf, col}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoders}

class TestFeaturesTransformer(override val uid: String) extends Transformer with DefaultParamsWritable {
  val testFile = new Param[String](this, "trainFile", "The path for the train file")

  def setTestFile(value: String): this.type = set(testFile, value)

  def this() = this(Identifiable.randomUID("features"))

  def copy(extra: ParamMap): TestFeaturesTransformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = Encoders.product[Features].schema

  def transform(df: Dataset[_]): DataFrame = {
    import df.sparkSession.implicits.newProductEncoder
    val spark = df.sparkSession
    val csvOptions = Map("header" -> "true", "escape" -> "\"")
    val testDF = spark.read.options(csvOptions).csv($(testFile))
    val hashUDF = udf((q: String) => q.hashCode)
    val typedTestDF = testDF
      .selectExpr("cast(test_id as int) id", "question1", "question2")
      .withColumn("isDuplicate", lit(false))
      .withColumn("qid1", hashUDF(col("question1")))
      .withColumn("qid2", hashUDF(col("question2")))
   val orderedTestDF =  typedTestDF.select("id", "qid1", "qid2", "question1", "question2", "isDuplicate")
   orderedTestDF.as[Features].toDF
  }
}
