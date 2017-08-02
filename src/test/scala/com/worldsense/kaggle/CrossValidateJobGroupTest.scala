package com.worldsense.kaggle

import java.io.File

import org.apache.commons.io.FileUtils
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

/*
import com.worldsense.indexer.JobGroupSpec
import com.worldsense.spark.HadoopUtils

// scalastyle:off magic.number
@RunWith(classOf[JUnitRunner])
class CrossValidateJobGroupTest extends JobGroupSpec {
  val cleanResource = new File(getClass.getResource("/quora/clean.json").getFile)
  val clean = new File(HadoopUtils.appendPath(datadir, cleanResource.getName))
  val modelDir = HadoopUtils.appendPath(datadir, "model")
  val estimatorDir = HadoopUtils.appendPath(datadir, "estimator")
  def populateDatadir() = {
    FileUtils.copyFile(cleanResource, clean)
  }
  // This test is tagged as ignore because spark ml is too slow on circleci
  "CrossValidateFeaturesJobGroup" should "cross validate a small model" ignore {
    populateDatadir()
    //Use toy hyperparameters to favor running speed.
    new ModelJobGroup(
      datadir, clean.getName, modelDir, estimatorDir, spark,
      stopwords = Nil, numTopics = 3, ldaMaxIterations = 2, vocabSize = 100, minDf = 2,
      lrMaxIterations = 2).run()
    val pipeline = Pipeline.load(estimatorDir)
    new CrossValidateJobGroup(
      datadir, clean.getName, estimatorDir, "crossValidated", spark,
      vocabSize = Array(1, 100), minDF = Array(2), regularization = Array(0.2), numTopics = Array(10)).run()
    val cleanFeaturesDF = spark.read.json(clean.getAbsolutePath)
    val crossValidated = PipelineModel.load(HadoopUtils.appendPath(datadir, "crossValidated"))
    val predictions = crossValidated.transform(cleanFeaturesDF)
    assert(predictions.columns.contains("p"))
  }
}
*/
