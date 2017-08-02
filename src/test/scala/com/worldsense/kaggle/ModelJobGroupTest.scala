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
class ModelJobGroupTest extends JobGroupSpec {
  val cleanResource = new File(getClass.getResource("/quora/clean.json").getFile)
  val clean = new File(HadoopUtils.appendPath(datadir, cleanResource.getName))
  val modelDir = HadoopUtils.appendPath(datadir, "model")
  val estimatorDir = HadoopUtils.appendPath(datadir, "estimator")
  def populateDatadir() = {
    FileUtils.copyFile(cleanResource, clean)
  }
  // This test is tagged as ignore because spark ml is too slow on circleci
  "ModelJobGroup" should "train a quick model on clean data" ignore {
    populateDatadir()
    //Use toy hyperparameters to favor running speed.
    new ModelJobGroup(
      datadir, clean.getName, modelDir, estimatorDir, spark,
      stopwords = Nil, numTopics = 3, ldaMaxIterations = 2, vocabSize = 100, minDf = 2,
      lrMaxIterations = 2).run()
    val cleanFeaturesDF = spark.read.json(clean.getAbsolutePath)
    val estimator = Pipeline.load(estimatorDir)
    assert(Option(estimator).isDefined)
    estimator.fit(cleanFeaturesDF)
    val model2 = estimator.fit(cleanFeaturesDF)
    val model = PipelineModel.load(modelDir)
    val vectors = model.transform(cleanFeaturesDF)
    assert(vectors.columns.contains("question1_stopworded_tokens_vector_lda"))
    assert(vectors.columns.contains("question2_stopworded_tokens_vector_lda"))
  }
}
*/
