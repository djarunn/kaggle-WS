package com.worldsense.kaggle
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{Estimator, Model, Transformer}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

trait GloveParams extends Params {
  final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  final def getInputCol: String = $(inputCol)
  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")
  final def getOutputCol: String = $(outputCol)
  final val sentenceLength: Param[Int] = new Param[Int](this, "sentenceLength", "output column name")
  final def getSentenceLength: Int = $(sentenceLength)
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setSentenceLength(value: Int): this.type = set(sentenceLength, value)
  setDefault(outputCol, uid + "__output")
  setDefault(inputCol, uid + "__input")
  setDefault(sentenceLength, 25)
}

class GloveEstimator(override val uid: String) extends Estimator[GloveModel] with GloveParams with DefaultParamsWritable {
  val vectorsPath: Param[String] = new Param(this, "vectorsPath", "path for the file with the vectors")
  def setVectorsPath(value: String): this.type = set(vectorsPath, value)
  setDefault(vectorsPath, "/tmp/news20/glove.6B/glove.6B.100d.txt")
  val normalizer: Param[Transformer] = new Param(this, "normalize", "normalize tokens")
  def this() = this(Identifiable.randomUID("word2vec"))
  def copy(extra: ParamMap): GloveEstimator = defaultCopy(extra)

  def outputDataType: DataType = VectorType
  protected def validateInputType(inputType: DataType): Unit = {
    require(inputType == ArrayType(StringType, containsNull = true), s"Input type must be tokens but got $inputType.")
  }
  override def transformSchema(schema: StructType): StructType = {
    val inputType = schema($(inputCol)).dataType
    validateInputType(inputType)
    if (schema.fieldNames.contains($(outputCol))) {
      throw new IllegalArgumentException(s"Output column ${$(outputCol)} already exists.")
    }
    val outputFields = schema.fields :+
      StructField($(outputCol), outputDataType, nullable = false)
    StructType(outputFields)
  }
  override def fit(dataset: Dataset[_]): GloveModel = {
    val vectors = GloveEstimator.load(dataset.sparkSession, $(vectorsPath))
    assert(vectors.values.map(_.length).toSeq.distinct.length == 1)
    new GloveModel(uid, vectors)
      .setParent(this)
      .setInputCol($(inputCol))
      .setOutputCol($(outputCol))
      .setSentenceLength($(sentenceLength))
  }
}

object GloveEstimator extends DefaultParamsReadable[GloveEstimator] {
  def load(spark: SparkSession, path: String): Map[String, Array[Float]] = {
    import spark.implicits._
    val vectors = spark.read.text(path).as[String].map { line =>
      val values = line.split(" ")
      val word = values(0)
      val coefs = values.slice(1, values.length).map(_.toFloat)
      (word, coefs)
    }
    vectors.collect.toMap
  }
}

class GloveModel(override val uid: String, private val word2vec: Map[String, Array[Float]]) extends Model[GloveModel] with GloveParams {
  assert(word2vec.values.map(_.length).toSeq.distinct.length == 1)
  private val dimensions = word2vec.headOption.map(_._2.length).getOrElse(0)
  protected def outputDataType: DataType = VectorType
  protected def validateInputType(inputType: DataType) = {}  // todo(implement me)
  override def transformSchema(schema: StructType): StructType = {
    val inputType = schema($(inputCol)).dataType
    validateInputType(inputType)
    if (schema.fieldNames.contains($(outputCol))) {
      throw new IllegalArgumentException(s"Output column ${$(outputCol)} already exists.")
    }
    val outputFields = schema.fields :+
      StructField($(outputCol), outputDataType, nullable = false)
    StructType(outputFields)
  }
  override def transform(ds: Dataset[_]): DataFrame = {
    val gloveBC = ds.sparkSession.sparkContext.broadcast(word2vec)
    val vectorizeUDF = udf((tokens: Seq[String]) => {
      val vectors = GloveModel.vectorize(gloveBC.value, tokens, $(sentenceLength))
      assert(vectors.flatten.length == $(sentenceLength) * 100)
      Vectors.dense(vectors.flatten.map(_.toDouble).toArray)
    }, outputDataType)
    ds.withColumn($(outputCol), vectorizeUDF(ds($(inputCol))))
  }
  def copy(extra: ParamMap): GloveModel = {
    new GloveModel(uid, word2vec)
      .setInputCol($(inputCol))
      .setOutputCol($(outputCol))
  }
}

object GloveModel {
  def vectorize(word2vec: Map[String, Array[Float]], tokens: Seq[String], padLength: Int): Seq[Array[Float]] = {
    val dimensions = word2vec.values.head.length
    val zeros = Array.fill[Float](dimensions)(0)
    val ones = Array.fill[Float](dimensions)(1)
    val vectors = tokens.map(token => word2vec.getOrElse(token, zeros))
    vectors.padTo(padLength, ones).take(padLength)
  }
}
