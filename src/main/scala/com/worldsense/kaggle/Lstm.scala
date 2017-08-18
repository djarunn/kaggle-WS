package com.worldsense.kaggle

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType

class Lstm(override val uid: String) extends Estimator[DLModel[Float]] with DefaultParamsWritable  {
  def this() = this(Identifiable.randomUID("lstm"))
  final val labelCol: Param[String] = new Param[String](this, "inputCol", "comma separate input column names")
  final val featuresCol: Param[String] = new Param[String](this, "featuresCol", "comma separate input column names")
  final val predictionCol: Param[String] = new Param[String](this, "predictionCol", "comma separate input column names")
  final val embeddingDim: Param[Int] = new Param[Int](this, "outputCol", "comma separate input column names")
  setDefault(embeddingDim, 100)
  final val paddingLength: Param[Int] = new Param[Int](this, "paddingLength", "comma separate input column names")
  setDefault(paddingLength, 20)
  final val batchSize: Param[Int] = new Param[Int](this, "batchSize", "comma separate input column names")
  setDefault(batchSize, 16)   // needs to be multiple of core numbers
  final val hiddenDim: Param[Int] = new Param[Int](this, "hiddenDim", "comma separate input column names")
  setDefault(hiddenDim, 32)
  final val maxEpoch: Param[Int] = new Param[Int](this, "maxEpoch", "comma separate input column names")
  setDefault(maxEpoch, 100)
  final def getLabelCol: String = $(labelCol)
  final def getFeaturesCol: String = $(featuresCol)
  final def getPredictionCol: String = $(predictionCol)
  final def getEmbeddingDim: Int = $(embeddingDim)
  final def getPaddingLength: Int = $(paddingLength)
  final def getBatchSize: Int = $(batchSize)
  final def getMaxEpoch: Int = $(maxEpoch)
  final def setLabelCol(col: String): this.type = set(labelCol, col)
  final def setFeaturesCol(col: String): this.type = set(featuresCol, col)
  final def setPredictionCol(col: String): this.type = set(predictionCol, col)
  final def setEmbeddingDim(value: Int): this.type = set(embeddingDim, value)
  final def setPaddingLength(value: Int): this.type = set(paddingLength, value)
  final def setBatchSize(value: Int): this.type = set(batchSize, value)
  final def setHiddenDim(value: Int): this.type = set(hiddenDim, value)
  final def setMaxEpoch(value: Int): this.type = set(maxEpoch, value)

  override def transformSchema(schema: StructType): StructType = {
    assembleNeuralNetwork().transformSchema(schema)
  }
  override def fit(dataset: Dataset[_]): DLModel[Float] = assembleNeuralNetwork().fit(dataset)
  private def assembleNeuralNetwork(): DLEstimator[Float] = {
    val numClasses = 2   // simplifies code and we do no need to support multiclass
    // The input vector for the neural network has batchSize x Padding x Dimension length, and is assembled
    // from the dataset with rows of Padding x Dimension float 1d vectors.
    val nn: Sequential[Float] = new Sequential[Float]()
      //.add(Padding(1, $(paddingLength), 2))
      .add(Recurrent[Float]()
         .add(LSTM($(embeddingDim), $(hiddenDim))))
      .add(Select(2, -1))
      .add(Linear($(hiddenDim), numClasses))
      .add(SoftMax())
    val criterion: Criterion[Float] = new ClassNLLCriterion[Float]()
    val estimator = new DLClassifier[Float](nn, criterion, Array($(paddingLength), $(embeddingDim)))
    estimator.setFeaturesCol($(featuresCol))
    estimator.setLabelCol($(labelCol))
    estimator.setPredictionCol($(predictionCol))
    estimator.setBatchSize($(batchSize))
    estimator.setMaxEpoch($(maxEpoch))
    estimator
  }
  override def copy(extra: ParamMap): Lstm = defaultCopy(extra)
}
object Lstm extends DefaultParamsReadable[Lstm]
