import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.util.MLUtils

// Import relevant libraries
import org.apache.spark.ml.feature.NGram
import org.apache.spark.sql.functions._
// imports for the text document pipeline
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{Tokenizer, StopWordsRemover}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}

// decision tree imports
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegressionModel

// importing CSV data into the expected format
import org.apache.spark.mllib.regression.LabeledPoint
//import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.sql.Row

// for assembler
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.{HashingTF, IDF}

import org.apache.spark.sql.types.IntegerType

import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}

import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

// Define main method (scala entry point)
object ExerciseRegHigh{

  def main(args: Array[String]){

        // Create Spark session
        val sparkSession = SparkSession.builder
            .master("local")
            .appName("ExerciseRegHigh")
            .getOrCreate()
            
        // Step 1
        val sc = sparkSession.sparkContext
              
        // Import implicits
        import sparkSession.implicits._
        
        // Step 3 Part 1: Get the execution time
        val time = System.nanoTime
        
        // Load the data
        val higgs = sc.textFile("HIGGS_r.csv.gz")
        
        // Create dataframe
        val data = higgs.map(line => line.split(",").map(_.toDouble)).toDF()
        //println(data.rdd.partitions.size)
        
        // Create columns
        val data_map = data.select($"value"(0) as "label", $"value"(1) as "_c1", $"value"(2) as "_c2",
                                   $"value"(3) as "_c3", $"value"(4) as "_c4", $"value"(5) as "_c5",
                                   $"value"(6) as "_c6", $"value"(7) as "_c7", $"value"(8) as "_c8",
                                   $"value"(9) as "_c9", $"value"(10) as "_c10", $"value"(11) as "_c11",
                                   $"value"(12) as "_c12", $"value"(13) as "_c13", $"value"(14) as "_c14",
                                   $"value"(15) as "_c15", $"value"(16) as "_c16", $"value"(17) as "_c17",
                                   $"value"(18) as "_c18", $"value"(19) as "_c19", $"value"(20) as "_c20",
                                   $"value"(21) as "_c21", $"value"(22) as "_c22", $"value"(23) as "_c23",
                                   $"value"(24) as "_c24", $"value"(25) as "_c25", $"value"(26) as "_c26",
                                   $"value"(27) as "_c27", $"value"(28) as "_c28")
        
        //data_map.show()
        //data.show()
        //data.printSchema
        
        // Create assembler for each algorithm
        // Put assembler & model in pipeline
        
        // Split the data into training and test sets (30% held out for testing)
        val splits = data_map.randomSplit(Array(0.7, 0.3))
        val (trainingData, testData) = (splits(0), splits(1))
        
        //trainingData.show()
        //testData.show()
        
        // Assembler for classification
        val assembler_class = new VectorAssembler()
        // take out "_c1" - "_c21" to only have high level features
        // "_c1", "_c2", "_c3", "_c4", "_c5", "_c6", "_c7", "_c8", "_c9", "_c10", "_c11", 
        // "_c12", "_c13", "_c14", "_c15", "_c16", "_c17", "_c18", "_c19", "_c20", "_c21",
           .setInputCols(Array("_c22", "_c23", "_c24", "_c25", "_c26", "_c27", "_c28"))
           .setOutputCol("features")
        //val output_new = assembler_class.select("label", "features")
        //output_new.show()
        
        // Create Decision Tree Classifier
        //val decision_tree = new DecisionTreeClassifier()
        //  .setLabelCol("label")
        //  .setFeaturesCol("features")
        
        // Create Logistic or Regression Tree Classifier
        val decision_tree = new DecisionTreeRegressor()
        // change to LogisticRegression() for logistic regression 
        // and DecisionTreeRegressor() for regression
            .setLabelCol("label")
            .setFeaturesCol("features")
        
        // Pipeline
        val pipeline_class = new Pipeline()
            .setStages(Array(assembler_class, decision_tree))
        
        // Param Grid
        val paramGrid = new ParamGridBuilder()
          .addGrid(decision_tree.impurity, Array("variance"))
        // change to decision_tree.impurity, Array("variance") for regression
        // and decision_tree.impurity, Array("entropy", "gini") for classification
        // and comment out for logistic regression
        //  .addGrid(decision_tree.maxIter, Array(1, 30, 50))
        //  .addGrid(decision_tree.regParam, Array(0.1, 0.5, 1))
        //  .addGrid(decision_tree.threshold, Array(0.25, 0.5, 0.75))
        // the following is for classification and regression only:
          .addGrid(decision_tree.maxDepth, Array(1, 5, 10))
          .addGrid(decision_tree.maxBins, Array(5, 10, 20))
          .build()
        
        // Create evaluator
        val evaluator = new RegressionEvaluator() 
        // change to BinaryClassificationEvaluator() for classification and logistic regression
        // change to RegressionEvaluator() for regression
          .setLabelCol("label")
        
        // Cross Validator
        val cv = new CrossValidator()
          .setEstimator(pipeline_class)
          .setEvaluator(evaluator)
          .setEstimatorParamMaps(paramGrid)
          .setNumFolds(3)  // Use 3+ in practice
        
        // Run cross-validation, and choose the best set of parameters.
        val cvModel = cv.fit(trainingData)
        
        // Obtain the best parameters
        val best_model = cvModel.bestModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[DecisionTreeRegressionModel]
        // change .asInstanceOf[DecisionTreeClassificationModel] for classification and change .asInstanceOf[DecisionTreeRegressionModel]
        // for regression and .asInstanceOf[LogisticRegressionModel]
        println("Best Max Depth = " + best_model.parent.asInstanceOf[DecisionTreeRegressor].getMaxDepth)
        // use .asInstanceOf[DecisionTreeClassifier] for classification and .asInstanceOf[DecisionTreeRegressor] for regression
        // and .asInstanceOf[LogisticRegression] for logistic regression
        //use .getMaxDepth for classification and regression
        //use .getMaxIter for logistic regression
        println("Best Impurity = " + best_model.parent.asInstanceOf[DecisionTreeRegressor].getImpurity)
        //.getImpurity for classification and regression
        //.getRegParam for logistic regression
        println("Best Max Bins = " + best_model.parent.asInstanceOf[DecisionTreeRegressor].getMaxBins)
        //.getMaxBins for classification and regression
        //.getThreshold for logistic regression
        
        // Step 3 Part 2: Get the execution time
        val duration = (System.nanoTime - time) / 1e9d
        println("The execution time is " + duration + "seconds.")
        
        // Make predictions with test data
        val result_class = cvModel.transform(testData)
        //result_class.select("label", "probability", "prediction")
        
        // Get accuracy
        val accuracy = evaluator.evaluate(result_class)
        println("Accuracy = " + accuracy)
        println("Test Error = " + (1.0 - accuracy))
        
        sparkSession.stop()
  }
}

















