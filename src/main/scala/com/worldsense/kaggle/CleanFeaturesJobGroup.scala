package com.worldsense.kaggle

/** Consumes a dataframe from RawFeaturesJobGroup and cleans the data on it.
 *
 * Besides simple data transformations and empty value handling, this cleans the text using
 * worldsense bow machinery, then rewrites the cleaner version of the
 * text as CleanFeaturesJobGroup.Features, which can be easily processed by subsequent pipelines.
 *
 * The cleaning strategy simply delegates to existing worldsense pipeline, which will parse the
 * questions as if they were HTML documents, generate BoW representations, and then assemble
 * then again.
 */

import org.apache.spark.sql.SparkSession

import com.worldsense.client.ThriftHtmlFragment
import com.worldsense.spark.SerializedSparkJobGroup
import com.worldsense.thrift.ThriftBowUtils

class CleanFeaturesJobGroup(pathPrefix: String, rawDir: String, cleanDir: String, spark: SparkSession)
  extends SerializedSparkJobGroup(
    "Convert raw dataframe into dataset with text features cleaned with worldsense BoW pipeline",
    pathPrefix, rawDir, cleanDir, spark) {
  override def jobCode(): Boolean = {
    import spark.implicits.newProductEncoder
    val rawFeaturesDF = loadDF(rawDir)
        .withColumnRenamed("is_duplicate", "isDuplicate")  // get rid of dash
        .withColumnRenamed("test_id", "testId")  // get rid of dash
    val rawFeaturesDS = rawFeaturesDF.as[CleanFeaturesJobGroup.Features]
    val cleanFeaturesDS = rawFeaturesDS map { features =>
      features.copy(
        question1 = CleanFeaturesJobGroup.cleanQuestion(features.qid1, features.question1),
        question2 = CleanFeaturesJobGroup.cleanQuestion(features.qid2, features.question2))
    }
    cleanFeaturesDS.write.parquet(fullPath(pathPrefix, cleanDir))
    true
  }
}

object CleanFeaturesJobGroup {
  private[kaggle] def cleanQuestion(qid: BigInt, question: String): String = {
    val fragment = new ThriftHtmlFragment()
    fragment.setHtml(Option(question).getOrElse(""))
    fragment.setUrl(s"http://www.worldsense.com/kaggle/quora/$qid")
    val bow = ThriftBowUtils.html2bow(fragment)
    ThriftBowUtils.bow2text(bow)
  }

  case class Features(
      id: BigInt, qid1: BigInt, qid2: BigInt, question1: String, question2: String,
      isDuplicate: Boolean, source: String, testId: String)
}