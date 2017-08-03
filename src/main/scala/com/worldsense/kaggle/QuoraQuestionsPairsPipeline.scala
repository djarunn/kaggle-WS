package com.worldsense.kaggle

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType


class QuoraQuestionsPairsPipeline(override val uid: String) extends Estimator[PipelineModel] {
  def this() = this(Identifiable.randomUID("quoraquestionspairspipeline"))

  val tokenizerParam: Param[Tokenizer] = new Param(this, "tokenizer", "estimator for selection")
  def setTokenizer(value: Tokenizer): this.type = set(tokenizerParam, value)
  setDefault(tokenizerParam, new Tokenizer)

  val stopwordsRemoverParam: Param[StopWordsRemover] = new Param(this, "stopwords", "estimator for selection")
  def setStopwordsRemover(value: StopWordsRemover): this.type = set(stopwordsRemoverParam, value)
  setDefault(stopwordsRemoverParam, new StopWordsRemover())

  val countVectorizerParam: Param[CountVectorizer] = new Param(this, "countVectorizer", "estimator for selection")
  def setCountVectorizer(value: CountVectorizer): this.type = set(countVectorizerParam, value)
  setDefault(countVectorizerParam, new CountVectorizer())

  val ldaParam: Param[LDA] = new Param(this, "lda", "estimator for selection")
  def setLDA(value: LDA): this.type = set(ldaParam, value)
  setDefault(ldaParam, new LDA())

  val logisticRegressionParam: Param[LogisticRegression] = new Param(this, "logisticRegression", "estimator for selection")
  def setLogisticRegression(value: LogisticRegression): this.type = set(logisticRegressionParam, value)
  setDefault(logisticRegressionParam, new LogisticRegression())

  override def transformSchema(schema: StructType): StructType = assemblePipeline().transformSchema(schema)
  override def fit(dataset: Dataset[_]): PipelineModel = assemblePipeline().fit(dataset)
  def copy(extra: ParamMap): QuoraQuestionsPairsPipeline = defaultCopy(extra)

  private def assemblePipeline(): Pipeline = {
    val questions = Array("question1", "question2")
    val tokenized = questions.map(q => s"${q}_stopworded_tokens")
    val vectorized = tokenized.map(q => s"${q}_vector")
    val ldaed = vectorized.map(q => s"${q}_lda")

    val tokenizer = tokenizePipeline(questions)
    val vectorizer = vectorizePipeline(tokenized)
    val lda = ldaPipeline(vectorized)
    val lr = probabilityPipeline(ldaed)
    val stages = Array(tokenizer, vectorizer, lda, lr).flatten
    new Pipeline().setStages(stages)
  }
  def tokenizePipeline(columns: Array[String]): Array[PipelineStage] = {
    val mcTokenizer = new MultiColumnPipeline()
        .setStage($(tokenizerParam))
        .setInputCols(columns)
        .setOutputCols(columns.map(_ + "_tokens"))
    val mcStopwordsRemover = new MultiColumnPipeline()
        .setStage($(stopwordsRemoverParam))
        .setInputCols(mcTokenizer.getOutputCols)
        .setOutputCols(columns.map(_ + "_stopworded_tokens"))
    Array(mcTokenizer, mcStopwordsRemover)
  }

  def vectorizePipeline(columns: Array[String]): Array[PipelineStage] = {
    val mcCountVectorizer = new MultiColumnPipeline()
      .setInputCols(columns)
      .setOutputCols(columns.map(_ + "_vector"))
      .setStage($(countVectorizerParam))
    Array(mcCountVectorizer)
  }

 def ldaPipeline(columns: Array[String]): Array[PipelineStage] = {
   // The "em" optimizer supports serialization, is disk hungry and slow,
   // "online" is fast but cannot be serialized. We keep the latter as default, since this
   // model is only used to create a submission and nothing else.
   val optimizer = "online"
   val lda = $(ldaParam)
     .setOptimizer(optimizer)
     .setFeaturesCol("tmpinput").setTopicDistributionCol("tmpoutput")
   val mcLda = new MultiColumnPipeline()
     .setInputCols(columns)
     .setOutputCols(columns.map(_ + "_lda"))
     .setStage(lda, lda.getFeaturesCol, lda.getTopicDistributionCol)
   Array(mcLda)
  }

  def probabilityPipeline(columns: Array[String]): Array[PipelineStage] = {
    val labelCol = "isDuplicateLabel"
    val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("mergedlda")
    val labeler = new SQLTransformer().setStatement(
      s"SELECT *, cast(isDuplicate as int) $labelCol from __THIS__")
    val lr = new LogisticRegression()
        .setFeaturesCol("mergedlda").setProbabilityCol("p").setRawPredictionCol("raw")
        .setLabelCol(labelCol)
    Array(assembler, labeler, lr)
  }
}
