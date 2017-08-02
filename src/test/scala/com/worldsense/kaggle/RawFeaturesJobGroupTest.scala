package com.worldsense.kaggle

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

/*
// scalastyle:off magic.number
@RunWith(classOf[JUnitRunner])
class RawFeaturesJobGroupTest extends JobGroupSpec {
  val trainFile = getClass.getResource("/quora/train100.csv").getFile
  val testFile = getClass.getResource("/quora/test100.csv").getFile
  val quoraDir = HadoopUtils.appendPath(datadir, "quora")
  val rawDir = HadoopUtils.appendPath(datadir, "raw")
  def populateDatadir() = {
    val dstTrainFile = HadoopUtils.appendPath(quoraDir, "train.csv")
    HadoopUtils.copyFile(trainFile, dstTrainFile)
    val dstTestFile = HadoopUtils.appendPath(quoraDir, "test.csv")
    HadoopUtils.copyFile(testFile, dstTestFile)
  }
  "RawFeaturesJobGroup" should "extract all rows from input" in {
    populateDatadir()
    new RawFeaturesJobGroup(datadir, "quora", "raw", spark).run()
    val rawFeatures = spark.read.parquet(rawDir)
    assert(rawFeatures.count() === 2*100)
  }
}
*/
