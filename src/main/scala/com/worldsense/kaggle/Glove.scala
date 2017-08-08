package com.worldsense.kaggle
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{Estimator, Model, Transformer}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

trait GloveParams extends Params {
  final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  final def getInputCol: String = $(inputCol)
  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  setDefault(outputCol, uid + "__output")
}

class GloveEstimator(override val uid: String) extends Estimator[GloveModel] with GloveParams with DefaultParamsWritable {
  val vectorsPath: Param[String] = new Param(this, "vectorsPath", "path for the file with the vectors")
  def setVectorsPath(value: String): this.type = set(vectorsPath, value)
  val normalizer: Param[Transformer] = new Param(this, "normalize", "normalize tokens")
  def this() = this(Identifiable.randomUID("word2vec"))
  def copy(extra: ParamMap): GloveEstimator = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema
  override def fit(dataset: Dataset[_]): GloveModel = {
    val vectors = GloveEstimator.load(dataset.sparkSession, $(vectorsPath))
    new GloveModel(uid, vectors)
      .setParent(this)
      .setInputCol($(inputCol))
      .setOutputCol($(outputCol))
  }
}

object GloveEstimator extends DefaultParamsReadable[GloveEstimator] {
  def load(spark: SparkSession, path: String): Map[String, Array[Double]] = {
    import spark.implicits._
    val vectors = spark.read.text(path).as[String].map { line =>
      val values = line.split(" ")
      val word = values(0)
      val coefs = values.slice(1, values.length).map(_.toDouble)
      (word, coefs)
    }
    vectors.collect.toMap
  }
}

class GloveModel(override val uid: String, private val word2vec: Map[String, Array[Double]]) extends Model[GloveModel] with GloveParams {
  assert(word2vec.values.map(_.length).toSeq.distinct.length == 1)
  private val dimensions = word2vec.headOption.map(_._2.length).getOrElse(0)
  protected def outputDataType: DataType = new ArrayType(ArrayType(DoubleType, false), false)
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
    schema
  }
  override def transform(ds: Dataset[_]): DataFrame = {
    val gloveBC = ds.sparkSession.sparkContext.broadcast(word2vec)
    val vectorizeUDF = udf((tokens: Seq[String]) => tokens.map { token =>
      gloveBC.value.getOrElse(token, Array.fill[Double](dimensions)(0))
    }, outputDataType)
    ds.withColumn($(outputCol), vectorizeUDF(ds($(inputCol))))
  }
  def copy(extra: ParamMap): GloveModel = {
    new GloveModel(uid, word2vec)
      .setInputCol($(inputCol))
      .setOutputCol($(outputCol))
  }
}
