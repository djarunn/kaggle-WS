package com.worldsense.kaggle

import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.{CountVectorizer, SQLTransformer, StopWordsRemover, Tokenizer, VectorAssembler}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

import com.worldsense.spark.SerializedSparkJobGroup
class ModelJobGroup(pathPrefix: String, cleanDir: String, modelDir: String, estimatorDir: String,
    spark: SparkSession, stopwords: Seq[String],
    numTopics: Int, ldaMaxIterations: Int, vocabSize: Int, minDf: Int,
    lrMaxIterations: Int) extends SerializedSparkJobGroup(
      "Train a LDA/Logistic regression model for the kaggle quora challenge",
      pathPrefix, Vector(cleanDir), Vector(modelDir, estimatorDir), spark) {
  import spark.implicits.newProductEncoder
  override def jobCode(): Boolean = {
    val questions = Array("question1", "question2")
    val tokenized = questions.map(q => s"${q}_stopworded_tokens")
    val vectorized = tokenized.map(q => s"${q}_vector")
    val ldaed = vectorized.map(q => s"${q}_lda")

    val tokenizer = tokenizePipeline(questions)
    val vectorizer = vectorizePipeline(tokenized)
    val lda = ldaPipeline(vectorized)
    val lr = probabilityPipeline(ldaed)

    val pipeline = new Pipeline().setStages(Seq(tokenizer, vectorizer, lda, lr).flatten.toArray)

    val cleanDF = loadDF(cleanDir)
    val trainDF = cleanDF.filter(col("source") === "train")
    val model = pipeline.fit(trainDF)
    // Make predictions. Notice we use trainDF again, since test has no labels.
    val predictions = model.transform(trainDF).select("p", "isDuplicateLabel").as[(DenseVector, Int)]
    val predictionsAndLabels = predictions map { case (p, label) =>
      (p.values.last, label.toDouble)
    }
    val metrics = new BinaryClassificationMetrics(predictionsAndLabels.rdd)
    val areaUnderPR = metrics.areaUnderPR()
    val areaUnderROC = metrics.areaUnderROC()
    logger.info(s"trained a model with areaUnderPR $areaUnderPR and areaUnderROC $areaUnderROC")
    model.save(fullPath(pathPrefix, "model"))
    pipeline.save(fullPath(pathPrefix, "estimator"))
    true
  }
  private val logger = org.log4s.getLogger
  def tokenizePipeline(columns: Array[String]): Array[PipelineStage] = {
    val tokenizer = new MultiColumnPipeline()
        .setStage(new Tokenizer())
        .setInputCols(columns).setOutputCols(columns.map(_ + "_tokens"))
    val stopwordsRemover = new MultiColumnPipeline()
        .setStage(new StopWordsRemover().setStopWords(stopwords.toArray))
        .setInputCols(tokenizer.getOutputCols).setOutputCols(columns.map(_ + "_stopworded_tokens"))
    Array(tokenizer, stopwordsRemover)
  }

  def vectorizePipeline(columns: Array[String]): Array[PipelineStage] = {
    Array(new MultiColumnPipeline()
        .setInputCols(columns).setOutputCols(columns.map(_ + "_vector"))
        .setStage(new CountVectorizer().setMinDF(minDf).setVocabSize(vocabSize)))
  }

 def ldaPipeline(columns: Array[String]): Array[PipelineStage] = {
    // The "em" optimizer supports serialization, is disk hungry and slow,
    // "online" is fast but cannot be serialized. We keep the latter as default, since this
    // model is only used to create a submission and nothing else.
    val optimizer = "online"
    val lda = new LDA()
      .setOptimizer(optimizer).setK(numTopics).setMaxIter(ldaMaxIterations)
      .setFeaturesCol("tmpinput").setTopicDistributionCol("tmpoutput")
    Array(new MultiColumnPipeline()
      .setInputCols(columns).setOutputCols(columns.map(_ + "_lda"))
      .setStage(lda, lda.getFeaturesCol, lda.getTopicDistributionCol))
  }

  def probabilityPipeline(columns: Array[String]): Array[PipelineStage] = {
    val labelCol = "isDuplicateLabel"
    val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("mergedlda")
    val labeler = new SQLTransformer().setStatement(
      s"SELECT *, cast(isDuplicate as int) $labelCol from __THIS__")
    val lr = new LogisticRegression()
        .setFeaturesCol("mergedlda").setProbabilityCol("p").setRawPredictionCol("raw")
        .setLabelCol(labelCol).setMaxIter(lrMaxIterations)
    Array(assembler, labeler, lr)
  }
}
