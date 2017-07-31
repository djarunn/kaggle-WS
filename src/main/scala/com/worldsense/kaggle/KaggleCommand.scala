package com.worldsense.kaggle

import scala.collection.mutable.ArrayBuffer

import com.codahale.metrics.MetricRegistry
import io.dropwizard.cli.ConfiguredCommand
import io.dropwizard.setup.Bootstrap
import net.sourceforge.argparse4j.impl.Arguments
import net.sourceforge.argparse4j.inf.{Namespace, Subparser}
import nl.grons.metrics.scala.InstrumentedBuilder
import org.apache.spark.sql.SparkSession
import org.scalactic.Requirements

import com.worldsense.basic.Yaml
import com.worldsense.mixer.Argparse4j.setDefaultScala
import com.worldsense.mixer.Mixer
import com.worldsense.spark.{SerializedSparkJobGroup, SparkConfiguration, StageParser}

class KaggleCommand(registry: MetricRegistry)
    extends ConfiguredCommand[Mixer.Config]("kaggle", "Trains and evaluate a model for Kaggle Quora pairs challenge.")
            with InstrumentedBuilder with Requirements {
  val metricRegistry = registry
  private val logger = org.log4s.getLogger

  override def configure(subparser: Subparser) {
    super.configure(subparser)

    subparser.addArgument("--master").setDefaultScala("local[*]").help(
      "Spark master")

    subparser.addArgument("--data-directory").setDefaultScala("/tmp/bonisoft/")
        .help("Directory prefix for all stages")
    val defaultMaxNgramSize = 8
    subparser.addArgument("--ngram-max-size").`type`(classOf[Integer])
        .setDefaultScala(new Integer(defaultMaxNgramSize))
        .help("Limits the maximum size of ngram to be computed.")
    subparser.addArgument("--overwrite").`type`(classOf[Boolean]).action(
      Arguments.storeTrue).help(
          "If set, delete overwrite existing directories with new output.")
    subparser.addArgument("--first-stage").setDefaultScala("quora:raw")
        .help("The point to start the indexer computation.")
    subparser.addArgument("--last-stage").setDefaultScala("clean,estimator:crossValidated")
        .help("The final desired point of the indexer computation.")

    subparser.addArgument("--stopwords-path")
        .setDefaultScala("s3://worldsense/brdocs5/stopwords.txt")
        .help("Where within --data-directory to read stopwords from.")

    subparser.addArgument("--lda-topics").`type`(classOf[Integer])
        .setDefaultScala(new Integer(20))
        .help("Number of topics to infer for LDA.")

    subparser.addArgument("--lda-max-iterations").`type`(classOf[Integer])
        .setDefaultScala(new Integer(100))
        .help("Number of iterations of learning for LDA.")

    subparser.addArgument("--lda-vocabulary-size").`type`(classOf[Int])
        .setDefaultScala(new Integer(10000))
        .help("Number of distinct word types to use while training the LDA model.")

    subparser.addArgument("--lda-min-df").`type`(classOf[Int])
        .setDefaultScala(new Integer(2))
        .help("Minimum number of different documents a term must appear in to be included in the vocabulary.")

    subparser.addArgument("--logistic-regression-max-iterations").`type`(classOf[Int])
        .setDefaultScala(new Integer(100))
        .help("Maximum number of iterations for logistic regression.")
  }

  override protected def run(
      bootstrap: Bootstrap[Mixer.Config], ns: Namespace,
      conf: Mixer.Config) {

    val sparkModule = new SparkConfiguration.Module {
      override lazy val sparkConfig = conf.worldsense.spark

    }
    val spark = sparkModule.spark
    logger.info(s"Configuration:\n${Yaml.toYaml(sparkModule.sparkConfig)}")
    val jobGroups = populateJobs(ns.getString("data_directory"), ns, spark)

    val stageParser = new StageParser(jobGroups)
    val overwrite = ns.getBoolean("overwrite")
    val stagesExecuted = stageParser.execute(
      ns.getString("first_stage"), ns.getString("last_stage"), overwrite)

    logger.info(s"Indexing Pipeline completed with $stagesExecuted stages executed.")
    sparkModule.stop()
  }

  // Populates all the jobs that are to be sequentially run in the indexing pipeline.
  // scalastyle:off method.length
  private def populateJobs(
      datadir: String, ns: Namespace, spark: SparkSession): ArrayBuffer[SerializedSparkJobGroup] = {
    val jobGroups: ArrayBuffer[SerializedSparkJobGroup] = ArrayBuffer()

    val quoraDir = "quora"

    val rawDir = "raw"
    val quora2raw = new RawFeaturesJobGroup(datadir, quoraDir, rawDir, spark)
    jobGroups.append(quora2raw)

    val cleanDir = "clean"
    val raw2clean = new CleanFeaturesJobGroup(datadir, rawDir, cleanDir, spark)
    jobGroups.append(raw2clean)

    val stopwords = spark.read.textFile(ns.getString("stopwords_path")).collect()
    val modelDir = "model"
    val estimatorDir = "estimator"
    val clean2model: SerializedSparkJobGroup = new ModelJobGroup(
      datadir, cleanDir, modelDir, estimatorDir, spark,
      Nil, ns.getInt("lda_topics"), ns.getInt("lda_max_iterations"),
      ns.getInt("lda_vocabulary_size"), ns.getInt("lda_min_df"),
      ns.getInt("logistic_regression_max_iterations"))
    jobGroups.append(clean2model)

    val submissionDir = "submission"
    val model2submission: SerializedSparkJobGroup = new SubmissionJobGroup(
      datadir, cleanDir, modelDir, submissionDir, spark)
    jobGroups.append(model2submission)

    val crossValidatedDir = "crossValidated"
    val model2crossvalidated = new CrossValidateJobGroup(
      datadir, cleanDir, estimatorDir, crossValidatedDir, spark)
    jobGroups.append(model2crossvalidated)
    val cvsubmissionDir = "cvsubmission"
    val cv2submssion: SerializedSparkJobGroup = new SubmissionJobGroup(
      datadir, cleanDir, crossValidatedDir, cvsubmissionDir, spark)
    jobGroups.append(cv2submssion)

    jobGroups
  }
}
