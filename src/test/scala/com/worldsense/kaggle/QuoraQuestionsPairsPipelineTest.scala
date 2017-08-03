package com.worldsense.kaggle

import java.io.File
import java.nio.file.{Files, Path, Paths}

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.scalatest.FlatSpec

import scala.util.Random

class QuoraQuestionsPairsPipelineTest extends FlatSpec with DataFrameSuiteBase {
  val rng = new Random(1)
  val features = (1 until 100).map { i =>
    val isDuplicate = rng.nextBoolean()
    val question1 = if (rng.nextBoolean()) "dunno" else "either"
    val question2 = if (isDuplicate) "i am a dup" else "i am not"
    Features(id = i, qid1 = rng.nextInt(100), qid2 = rng.nextInt(100),
      question1 = question1, question2 = question2,
      isDuplicate = isDuplicate)
  }
  "QuoraQuestionsPairsPipeline" should "serialize back and forth" in {
    import spark.implicits._
    val tmpdir: Path = Files.createTempDirectory("qqpptest")

    val estimatorDir: Path = Paths.get(tmpdir.toString, "estimator")
    val estimator = new QuoraQuestionsPairsPipeline()
    val ds = spark.createDataset(features)
    val p = estimator.fit(ds).transform(ds)
    assert(p.count() === features.length)
    val dupCount = p.select("prediction").as[Double].collect().count(_ > 0)
    assert(dupCount >= 10 && dupCount <= 90)  // learned something

    new File(tmpdir.toUri).delete()
  }
}
