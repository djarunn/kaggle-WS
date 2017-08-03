package com.worldsense.kaggle

import java.io.File
import java.nio.file.{Files, Path, Paths}

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import com.worldsense.kaggle.FeaturesLoader.Features
import org.scalatest.FlatSpec

import scala.util.Random

class QuoraQuestionsPairsCrossValidatorTest extends FlatSpec with DataFrameSuiteBase {
  val rng = new Random(2)
  val features = (1 until 100).map { i =>
    val isDuplicate = rng.nextBoolean()
    val question1 = if (rng.nextBoolean()) "dunno" else "either"
    val question2 = if (isDuplicate) "i am a dup" else "i am not"
    Features(id = i, qid1 = rng.nextInt(100), qid2 = rng.nextInt(100),
      question1 = question1, question2 = question2,
      isDuplicate = isDuplicate)
  }
  "QuoraQuestionsPairsCrossValidator" should "test some params" in {
    import spark.implicits._
    val tmpdir: Path = Files.createTempDirectory("qqpcvtest")
    val cvDir: Path = Paths.get(tmpdir.toString, "cv")
    // Create cross validator with toy hyperparameters favoring speed of execution.
    val cv = new QuoraQuestionsPairsCrossValidator()
      .setVocabularySize(List(1, 100))
      .setMinDF(List(2))
      .setRegularization(List(0.2))
      .setNumTopics(List(10))
    val ds = spark.createDataset(features)
    val p = cv.fit(ds).transform(ds)
    assert(p.collect.nonEmpty)
    new File(tmpdir.toUri).delete()

  }
}
