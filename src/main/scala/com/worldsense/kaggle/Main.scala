package com.worldsense.kaggle

import com.github.fommil.netlib.BLAS
import net.sourceforge.argparse4j.ArgumentParsers
import net.sourceforge.argparse4j.inf.ArgumentParserException
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

import scala.util.{Failure, Try}

object Main extends App {
  private val logger = org.log4s.getLogger
  private val sparkConf = new SparkConf()
    .set("spark.driver.memory", "6g")
    .setMaster("local[*]")
  private val featuresLoader = new FeaturesLoader()
  private val parser = ArgumentParsers.newArgumentParser("kaggle")
    .description("Train and evaluate a model for kaggle's  quora questions pair challenge.")
  parser.addArgument("trainingDataFile")
    .help("The training data file provided by kaggle")
  parser.addArgument("testDataFile")
    .help("The test data file provided by kaggle")
  parser.addArgument("submissionFile")
    .help("The file to create with the results")
  Try(parser.parseArgs(args)) recoverWith { case e: ArgumentParserException =>
    parser.handleError(e)
    System.exit(1)
    Failure(e)
  } foreach { res =>
    val spark = SparkSession.builder.config(sparkConf).appName("kaggle").getOrCreate()
    logger.info(s"BLAS backend is ${BLAS.getInstance().getClass.getName}")
    run(spark, res.getString("trainingDataFile"), res.getString("testDataFile"), res.getString("submissionFile"))
    spark.stop
  }
  def run(spark: SparkSession, trainingDataFile: String, testDataFile: String, submissionFile: String): Unit = {
    val crossValidator = new QuoraQuestionsPairsCrossValidator
    logger.info(s"Cross validator params:\n${crossValidator.explainParams()}")
    val numVariations = crossValidator.extractParamMap().toSeq.map(_.value.asInstanceOf[List[_]].length).product
    logger.info(s"Cross validator for kaggle quora questions pairs will train $numVariations * ${crossValidator.numFolds} models")

    // Train with cross validation to get the best params.
    val trainData = featuresLoader.loadTrainFile(spark, trainingDataFile)
    trainData.cache()   // we will use this repeatedly
    val cvModel = crossValidator.fit(trainData)
    val bestParams = cvModel.getEstimatorParamMaps.zip(cvModel.avgMetrics).maxBy(_._2)._1

    // Train on all data."
    val estimator = new QuoraQuestionsPairsPipeline().copy(bestParams)
    val model = estimator.fit(trainData)
    logger.info(s"Trained final model with params:\n${bestParams.toString}")

    val testData = featuresLoader.loadTestFile(spark, testDataFile)
    val submissionWriter = new SubmissionWriter().setModel(model)
    submissionWriter.writeSubmissionFile(testData, submissionFile)
    logger.info(s"Submission file parquet written at submissionFile.")
  }
}
