package com.worldsense.kaggle

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import com.worldsense.kaggle.FeaturesLoader.Features
import org.apache.spark.ml.clustering.LDA
import org.scalatest.FlatSpec

import scala.util.Random

class QuoraQuestionsPairsPipelineTest extends FlatSpec with DataFrameSuiteBase {
  val rng = new Random(1)
  val features = (1 until 100).map { i =>
    val isDuplicate = rng.nextBoolean()
    val question1 = if (rng.nextBoolean()) "dunno" else "either"
    val question2 = if (rng.nextBoolean()) "noise" else if (isDuplicate) "i am a dup." else "i am not: doubled"
    Features(id = i, qid1 = rng.nextInt(100), qid2 = rng.nextInt(100),
      question1 = question1, question2 = question2,
      isDuplicate = isDuplicate)
  }
  "QuoraQuestionsPairsPipeline" should "train a simple model" in {
    import spark.implicits.{newDoubleEncoder, newProductEncoder}
    val estimator = new QuoraQuestionsPairsPipeline()
    // Tune LDA to make learning easier since vocab is very small
    estimator.setLDA(new LDA().setK(3))
    val ds = spark.createDataset(features)
    val p = estimator.fit(ds).transform(ds)
    assert(p.count() === features.length)
    val dupCount = p.select("prediction").as[Double].collect().count(_ > 0)
    assert(dupCount >= 1 && dupCount <= 99)  // learned something
  }
}
