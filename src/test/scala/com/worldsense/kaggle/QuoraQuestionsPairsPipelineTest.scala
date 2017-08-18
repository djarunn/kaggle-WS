package com.worldsense.kaggle

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import com.intel.analytics.bigdl.utils.Engine
import com.worldsense.kaggle.FeaturesLoader.Features
import org.apache.spark.SparkConf
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.linalg.Vector
import org.scalatest.FlatSpec

import scala.util.Random

class QuoraQuestionsPairsPipelineTest extends FlatSpec with DataFrameSuiteBase {
  override def conf: SparkConf = Engine.createSparkConf(super.conf.setMaster("local[*]"))
  override protected implicit def enableHiveSupport: Boolean = false
  val rng = new Random(1)
  val features = (0 until 64).map { i =>
    val isDuplicate = rng.nextBoolean()
    val question1 = if (rng.nextBoolean()) "dunno" else "either"
    val question2 = if (isDuplicate) "i'am a dup." else "i am not: doubled"
    Features(id = i, qid1 = rng.nextInt(100), qid2 = rng.nextInt(100),
      question1 = question1, question2 = question2,
      isDuplicate = isDuplicate)
  }
  "QuoraQuestionsPairsPipeline" should "train a simple model" in {
    Engine.init
    import spark.implicits.newProductEncoder
    val gloveFile = getClass.getResource("/glove/glove.6B.7d.sample.txt").getFile
    val estimator = new QuoraQuestionsPairsPipeline()
    // Tune LDA to make learning easier since vocab is very small
    estimator.setLDA(new LDA().setK(3))
    estimator
      .setGlove(new GloveEstimator().setVectorsPath(gloveFile).setSentenceLength(2))
      .setLstm(new Lstm().setBatchSize(8).setEmbeddingDim(7).setHiddenDim(8).setMaxEpoch(50))
    val ds = spark.createDataset(features)
    val m = estimator.fit(ds)
    val p = m.transform(ds)
    val (dup, not) = p.rdd.map(_.getAs[Vector]("p").toArray.head).collect().partition(_ > 0.5)
    assert(dup.length > 20 && not.length > 20)
  }
}
