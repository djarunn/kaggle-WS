package com.worldsense.kaggle

import java.io.File

import org.apache.commons.io.FileUtils
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{CountVectorizer, Tokenizer}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

import com.worldsense.indexer.JobGroupSpec
import com.worldsense.spark.HadoopUtils

// scalastyle:off magic.number
@RunWith(classOf[JUnitRunner])
class MultiColumnPipelineTest extends JobGroupSpec {
  val cleanResource = new File(getClass.getResource("/quora/clean.json").getFile)
  val clean = new File(HadoopUtils.appendPath(datadir, cleanResource.getName))
  val estimatorDir = HadoopUtils.appendPath(datadir, "estimator")
  def populateDatadir() = {
    FileUtils.copyFile(cleanResource, clean)
  }
  "MultiColumnCountVectorizer" should "serialize back and forth" in {
    populateDatadir()
    val tokenizer  = new MultiColumnPipeline()
        .setInputCols(Array("question1", "question2"))
        .setOutputCols(Array("question1toc", "question2toc"))
        .setStage(new Tokenizer)
    tokenizer.save(estimatorDir)
    val loaded = MultiColumnPipeline.read.load(estimatorDir)
    val cleanDF: DataFrame = spark.read.json(clean.getAbsolutePath)
    loaded.fit(cleanDF)
  }
  it should "transform quora input" in {
    populateDatadir()
    val cleanDF: DataFrame = spark.read.json(clean.getAbsolutePath)
    val tokenizer1 = new Tokenizer().setInputCol("question1").setOutputCol("question1tok")
    val tokenizer2 = new Tokenizer().setInputCol("question2").setOutputCol("question2tok")
    val cv = new CountVectorizer().setInputCol("tmpinput").setOutputCol("tmpoutput")
    val countVectorizer = new MultiColumnPipeline()
        .setInputCols(Array("question1tok", "question2tok"))
        .setOutputCols(Array("question1vec", "question2vec"))
        .setStage(cv, cv.getInputCol, cv.getOutputCol)
    val estimator = new Pipeline().setStages(Array(tokenizer1, tokenizer2, countVectorizer))
    val model = estimator.fit(cleanDF.filter(col("source") === "train"))
    val featuresDF = model.transform(cleanDF)
    assert(featuresDF.count() == cleanDF.count())
    estimator.write.save(estimatorDir)
    val estimator2 = Pipeline.load(estimatorDir)
    val model2 = estimator.fit(cleanDF.filter(col("source") === "train"))
    val featuresDF2 = model.transform(cleanDF)
    assert(featuresDF.count() == featuresDF2.count())
  }
}
