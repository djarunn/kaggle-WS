package com.worldsense.kaggle

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{DLEstimator, DLModel, Estimator}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType

class Lstm(override val uid: String) extends Estimator[DLModel[Float]] with DefaultParamsWritable  {
  def this() = this(Identifiable.randomUID("lstm"))
  final val labelCol: Param[String] = new Param[String](this, "inputCol", "comma separate input column names")
  final val featuresCol: Param[String] = new Param[String](this, "featuresCol", "comma separate input column names")
  final val predictionCol: Param[String] = new Param[String](this, "predictionCol", "comma separate input column names")
  final val embeddingDim: Param[Int] = new Param[Int](this, "outputCol", "comma separate input column names")
  setDefault(embeddingDim, 300)
  final val hiddenDim: Param[Int] = new Param[Int](this, "hiddenDim", "comma separate input column names")
  setDefault(hiddenDim, 128)
  final val numClasses: Param[Int] = new Param[Int](this, "numClasses", "comma separate input column names")
  setDefault(numClasses, 2)
  final def getLabelCol: String = $(labelCol)
  final def getFeaturesCol: String = $(featuresCol)
  final def getPredictionCol: String = $(predictionCol)
  final def getEmbeddingDim: Int = $(embeddingDim)
  final def setLabelCol(col: String) = set(labelCol, col)
  final def setFeaturesCol(col: String) = set(featuresCol, col)
  final def setPredictionCol(col: String) = set(predictionCol, col)
  final def setEmbeddingDim(value: Int) = set(embeddingDim, value)
  final def setHiddenDim(value: Int) = set(hiddenDim, value)
  final def setNumClasses(value: Int) = set(numClasses, value)

  override def transformSchema(schema: StructType): StructType = {
    assembleNeuralNetwork().transformSchema(schema)
  }
  override def fit(dataset: Dataset[_]): DLModel[Float] = assembleNeuralNetwork().fit(dataset)
  private def assembleNeuralNetwork(): DLEstimator[Float] = {
    val padding = 25
    val nn: Sequential[Float] = new Sequential[Float]()
      .add(Padding(2, padding, 3))
      .add(Recurrent[Float]()
        .add(LSTM($(embeddingDim), $(hiddenDim))))
      .add(Select(2, -1))
    val criterion: Criterion[Float] = new ClassNLLCriterion[Float]()
    val estimator = new DLEstimator[Float](nn, criterion, Array(1, $(embeddingDim)), Array($(numClasses)))
    estimator.setFeaturesCol($(featuresCol))
    estimator.setLabelCol($(labelCol))
    estimator.setPredictionCol($(predictionCol))
    estimator.setBatchSize(4)
    estimator
  }
  override def copy(extra: ParamMap): Lstm = defaultCopy(extra)
}
object Lstm extends DefaultParamsReadable[Lstm]
