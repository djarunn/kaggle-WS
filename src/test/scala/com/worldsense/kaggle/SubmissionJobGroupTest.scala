package com.worldsense.kaggle

import java.io.File

import org.apache.commons.io.FileUtils
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

import com.worldsense.indexer.JobGroupSpec
import com.worldsense.spark.HadoopUtils

// scalastyle:off magic.number
@RunWith(classOf[JUnitRunner])
class SubmissionJobGroupTest extends JobGroupSpec {
  val cleanResource = new File(getClass.getResource("/quora/clean.json").getFile)
  val clean = new File(HadoopUtils.appendPath(datadir, cleanResource.getName))
  val modelDir = HadoopUtils.appendPath(datadir, "model")
  val submissionDir = HadoopUtils.appendPath(datadir, "submission")
  def populateDatadir() = {
    FileUtils.copyFile(cleanResource, clean)
  }
  // This test is tagged as ignore because spark ml is too slow on circleci
  "SubmissionJobGroup" should "generate a valid submission file" ignore {
    populateDatadir()
    //Use toy hyperparameters to favor running speed.
    new ModelJobGroup(
      datadir, clean.getName, modelDir, "estimator", spark,
      stopwords = Nil, numTopics = 3, ldaMaxIterations = 2, vocabSize = 100, minDf = 2,
      lrMaxIterations = 2).run()
    new SubmissionJobGroup(
      datadir, clean.getName, modelDir, submissionDir, spark).run()
    val submission = spark.read.csv(submissionDir)
    assert(submission.count() == 100 + 1)  // rows plus header
  }
}
