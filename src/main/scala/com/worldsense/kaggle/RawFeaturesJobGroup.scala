package com.worldsense.kaggle
/** Read the data files from http://kaggle.com/c/quora-question-pairs.
 * The contents of both train and test files are loaded in a dataframe with minimal data transformations.
 */
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, lit}

class RawFeaturesJobGroup(trainFile: String, testFile: String, outputDir: String, spark: SparkSession) {
  override def jobCode(): Boolean = {
    val csvOptions = Map("header" -> "true", "escape" -> "\"")
    val trainDF = spark.read.options(csvOptions).csv(trainFile).withColumn("source", lit("train"))
    val testDF = spark.read.options(csvOptions).csv(testFile).withColumn("source", lit("test"))
    // We create a single schema for both tables by filling the missing columns,
    // which are all numeric, with 0. It is up to the consumer to understand which fields
    // have semantics and which do not.
    val allcols = trainDF.columns.union(testDF.columns).distinct
    val expandedTrainDF = allcols.diff(trainDF.columns).foldLeft(trainDF) { case (df, col) =>
      df.withColumn(col, lit(0))
    } select(allcols.map(col):_*)
    val expandedTestDF = allcols.diff(testDF.columns).foldLeft(testDF) { case (df, col) =>
      df.withColumn(col, lit(0))
    } select(allcols.map(col):_*)
    val typedDF = expandedTrainDF.union(expandedTestDF).selectExpr(
      "cast(id as int) id",
      "cast(qid1 as int) qid1",
      "cast(qid2 as int) qid2",
      "question1", "question2",
      "cast(is_duplicate as boolean) is_duplicate",
      "source",
      "cast(test_id as int) test_id"
    )
    typedDF.write.parquet(outputDir)
    true
  }
}
