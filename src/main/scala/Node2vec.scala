import breeze.linalg.DenseMatrix
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.mllib.random.UniformGenerator

import scala.util.Random

object   Node2vec {

  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder()
      .appName("Application")
      .config("spark.master", "local")
      .getOrCreate()

    val sc : SparkContext = spark.sparkContext

    val data: RDD[(Int, Int)] = read_data("./train_epin.csv",spark)


    data.take(10).foreach(x => print(x))

    val total_nodes = 40334
    val emb_dim = 50
    val epoch = 20
    val learning_rate = 1.0
    val batch_size = 10000
    val neg_samples = 20

    var new_mat_in = DenseMatrix.zeros[Float](emb_dim, total_nodes)
    var new_mat_out = DenseMatrix.zeros[Float](emb_dim, total_nodes)

    var emb_in = create_embedding_matrix(emb_dim, total_nodes)
    var emb_out = create_embedding_matrix(emb_dim, total_nodes)

// *********************************** epochs **********************
    val k: DenseMatrix[Float] = create_embedding_matrix(emb_dim,total_nodes)

     val seq : IndexedSeq[RDD[(Int, (Int, Int))]]  = create_batchs(data, batch_size)
     seq.foreach(rdd => for ((k,v) <- rdd.collect) {

      var emb_in_broadcast = sc.broadcast(emb_in)
      var emb_out_broadcast = sc.broadcast(emb_out)
      var (in_grads, out_grads ) = estimate_gradients_for_edge(v._1,v._2,emb_in_broadcast.value,emb_out_broadcast.value)

     val random = new Random()
      for(k <- 1 to 20){
        var (neg_in, neg_out) = estimate_gradients_for_negative_samples(in_grads._1,random.nextInt(40334),emb_in_broadcast.value,emb_out_broadcast.value)
         //print("leeeeeeenght",neg_in._2.length)
        in_grads._2 +=  neg_in._2
        out_grads._2 +=  neg_out._2

      }

      val in_grad_rdd = sc.parallelize(Array(in_grads).toSeq)
      val out_grad_rdd = sc.parallelize(Array(out_grads).toSeq)

      val in_grads_local = in_grad_rdd // has non unique keys
        .reduceByKey(_+_) // all keys are uique
        .collectAsMap() // dowload all the gradients to the driver and convert to HashMap (dictionary)

      val out_grads_local = out_grad_rdd
        .reduceByKey(_+_)
        .collectAsMap()

     for (k <- in_grads_local.keys) {
        new_mat_in(::,k) := in_grads_local(k)
       //print("3jeb",new_mat_in)
      }

       for (k <- out_grads_local.keys) {
         new_mat_out(::,k) := out_grads_local(k)
       }

       update_gradients(new_mat_in,new_mat_out,emb_in,emb_out,learning_rate)

    })

    //************************** End of gradient decent *****************
     //println("embin \n ", emb_in)
    estimate_neighbors(2,emb_in,emb_out)


  }
  def read_data(path: String, spark: SparkSession): RDD[(Int, Int)] = {
    spark.read.format("csv")
      // the original data is store in CSV format
      // header: source_node, destination_node
      // here we read the data from CSV and export it as RDD[(Int, Int)],
      // i.e. as RDD of edges
      .option("header", "true")
      // State that the header is present in the file
      .schema(StructType(Array(
        StructField("source_node", IntegerType, false),
        StructField("destination_node", IntegerType, false)
      )))
      // Define schema of the input data
      .load(path)
      // Read the file as DataFrame
      .rdd.map(row => (row.getAs[Int](0), row.getAs[Int](1)))
    // Interpret DF as RDD
  }


  def create_batchs(data : RDD[(Int,Int)], batche_size : Int):  IndexedSeq[RDD[(Int, (Int, Int))]]  ={
    //val data_size = data.map(l => l.toString().length).collect()
    val index = data.zipWithIndex()
    val new_data = index.map(x => (x._2,x._1))
    val nb_batch = (new_data.count()/ batche_size).toInt
    val batchs = for (i <- 0 until nb_batch)
      yield new_data.filter(i == _._1 / nb_batch).map(x => (i,x._2))
    //batchs.foreach(rdd => for ((k,v) <- rdd.collect) printf("key: %s, value: %s\n", k, v))
    return batchs
  }

  def create_embedding_matrix(emb_dim : Int, total_nodes: Int):DenseMatrix[Float] ={
    val rand = new UniformGenerator()
    val ind : Float = 1.0F
    val vals = Array.fill(emb_dim * total_nodes)(rand.nextValue().toFloat)
     val mat = new DenseMatrix(emb_dim,total_nodes, vals)
    return mat
  }

  def update_weights(grad_in : (Int, DenseVector[Float]), lr : Double,grad_out : (Int, DenseVector[Float]), emb_in: DenseMatrix[Float],
                       emb_out: DenseMatrix[Float]): Unit ={

     //emb_in -= grad_in* lr
     //emb_out -= grad_out * lr

  }

  def estimate_gradients_for_edge(
                                   source: Int,
                                   destination: Int,
                                   emb_in: DenseMatrix[Float],
                                   emb_out: DenseMatrix[Float]
                                 ) = {

    val in = emb_in(::, source)
    val out = emb_out(::, destination)
    val neg: Int  = -1
    val in_grad = out * ( sigmoid(in.t * out) * neg)
    val out_grad = in * ( sigmoid(in.t * out)* neg)
    // return a tuple
    // Tuple((Int, DenseVector), (Int, DenseVector))
    // this tuple contains sparse gradients
    // in_grads is vector of gradients for
    // a node with id = source
    //((source, in_grads), (destination, out_grads))
    ((source, in_grad),(destination, out_grad))
  }

  def estimate_gradients_for_negative_samples(
                                   source: Int,
                                   destination: Int,
                                   emb_in: DenseMatrix[Float],
                                   emb_out: DenseMatrix[Float]
                                 ) = {

    val in = emb_in(::, source)
    val out = emb_out(::, destination)

    val in_grad_neg = out * (1+ sigmoid(in.t * out))
    val out_grad_neg = in * ( 1 + sigmoid(in.t * out))

    // return a tuple
    // Tuple((Int, DenseVector), (Int, DenseVector))
    // this tuple contains sparse gradients
    // in_grads is vector of gradients for
    // a node with id = source
    //((source, in_grads), (destination, out_grads))
    ((source, in_grad_neg),(destination, out_grad_neg))
  }

  def update_gradients(mat_in: DenseMatrix[Float],mat_out:DenseMatrix[Float],emb_in: DenseMatrix[Float],
                       emb_out: DenseMatrix[Float], lr: Double): Unit ={

    //println("maaatin", mat_in(0,1))
    emb_in -= mat_in
    emb_out -= mat_out

  }

  def estimate_neighbors(node : Int, emb_in: DenseMatrix[Float],
                         emb_out: DenseMatrix[Float]): Unit ={
    var result = new Array[(Int,Float)](40333)
    var resultats = Map[Int, Float]()
 //   val result = emb_in.t * emb_out
    for(i <- 0 to 40332){
      if(i == node){

      }else{

         resultats += (i -> emb_in(::,node).t * emb_out(::,i))
      }

    }
    resultats
  }
}
