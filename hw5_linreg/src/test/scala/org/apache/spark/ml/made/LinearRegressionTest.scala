package org.apache.spark.ml.made

import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, SparkSession}
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.ml.made.LinearRegressionTest.{_data_hardcode, _hidden_model}



class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 2.0
  lazy val data_hardcode: DataFrame = LinearRegressionTest._data_hardcode
  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val test_coeff = DenseVector[Double](1.0, 0.2, 0.6, 1.0)


  "Model" should "fit input data" in {
    val model = new LinearRegression ()
      .setInputCol("features")
      .setOutputCol("targets")
      .setPredCol("preds")
      .setIter(10)
      .setLr(1e-2)


    val trained_model = model.fit(data_hardcode)
    trained_model.coeff.length should be(4)
    trained_model.coeff(0) should be((1.5) +- delta)
    trained_model.coeff(1) should be((0.3) +- delta)
    trained_model.coeff(2) should be((-0.7) +- delta)
    trained_model.coeff(3) should be((0.0) +- delta)

  }

  "Model" should "make predictions" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      coeff = test_coeff
    ).setInputCol("features")
      .setOutputCol("targets")
      .setPredCol("preds")

    val vectors: Array[Vector] = model.transform(data).collect().map(_.getAs[Vector](0))

    vectors.length should be(10)

  }


  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setOutputCol("targets")
        .setIter(10)
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

    val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    model.isInstanceOf[LinearRegressionModel] should be(true)
    model.coeff.length should be(4)
  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setIter(10)
//        .setOutputCol("targets")
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)


    val read_data = reRead.transform(data)

    read_data.first().length should be(10)

  }
}

object LinearRegressionTest extends WithSpark {

  // features is a matrix of num_of_rows x 3 . Target num_of_rows=1e5
  lazy val _features_hardcode = DenseMatrix(
    (0.3, 0.5, 0.1),
    (0.4, 1.0, 0.7),
    (0.5, 0.6, 0.7))
  lazy val _features = DenseMatrix.rand(10, 3)

  //default hidden model is [1.5, 0.3, -0.7]
  lazy val _hidden_model = DenseVector[Double](1.5, 0.3, -0.7)

  //target labels are features x hidden_model, [num_of_rows x 3]  * [3 x 1] = [num_of_rows x 1]
  lazy val _target = (_features * _hidden_model.asDenseMatrix.t).toDenseVector
  lazy val _target_hardcode = (_features_hardcode * _hidden_model.asDenseMatrix.t).toDenseVector

  //dataframe is {_features | _targets}
  lazy val _data: DataFrame = {
    import sqlc.implicits._
    Range(0, _features.rows)
      .map(x => Tuple2(Vectors.fromBreeze(_features(x, ::).t), _target(x)))
      .toSeq
      .toDF("features", "targets")
  }

  lazy val _data_hardcode: DataFrame = {
    import sqlc.implicits._
    Range(0, _features_hardcode.rows)
      .map(x => Tuple2(Vectors.fromBreeze(_features_hardcode(x, ::).t), _target_hardcode(x)))
      .toSeq
      .toDF("features", "targets")
  }

}
