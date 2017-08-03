/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.feature

// Please see https://github.com/apache/spark/pull/12614

import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasInputCols
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

/**
 * :: Experimental ::
 * Utility transformer for removing columns from a DataFrame.
 */
@Experimental
class ColumnPruner (override val uid: String)
  extends Transformer with HasInputCols with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("columnPruner"))

  /** @group setParam */
  def setInputCols(values: Array[String]): this.type = set(inputCols, values)

  @Since("2.0.0")
  override def transform(dataset: Dataset[_]): DataFrame = {
    checkCanTransform(dataset.schema)
    val columnsToKeep = dataset.columns.filter(!$(inputCols).contains(_))
    dataset.select(columnsToKeep.map(dataset.col): _*)
  }

  override def transformSchema(schema: StructType): StructType = {
    checkCanTransform(schema)
    StructType(schema.fields.filter(col => !$(inputCols).contains(col.name)))
  }

  private def checkCanTransform(schema: StructType): Unit = {
    require(get(inputCols).isDefined, "Input cols must be defined first.")
    require($(inputCols).distinct.length == $(inputCols).length, "Input cols must be distinct.")
  }

  override def copy(extra: ParamMap): ColumnPruner = defaultCopy[ColumnPruner](extra)
}

@Since("2.0.0")
object ColumnPruner extends DefaultParamsReadable[ColumnPruner] {
  @Since("2.0.0")
  override def load(path: String): ColumnPruner = super.load(path)
}
