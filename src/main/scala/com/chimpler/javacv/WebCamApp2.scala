package com.chimpler.javacv

import java.awt.image.BufferedImage

import org.bytedeco.javacpp.helper.opencv_core.AbstractCvScalar
import org.bytedeco.javacpp.opencv_core._
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier
import org.bytedeco.javacpp.{opencv_imgproc, opencv_core}
import org.bytedeco.javacv.FrameGrabber.ImageMode
import org.bytedeco.javacv.{OpenCVFrameGrabber, CanvasFrame}
import scala.collection.mutable

/**
 * Created by frederic on 7/13/14.
 */
object WebCamApp2 extends App {
  val canvas = new CanvasFrame("Webcam")

  //  //Set Canvas frame to close on exit
  canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE)

  //Declare FrameGrabber to import output from webcam
  val grabber = new OpenCVFrameGrabber(0)
  grabber.setImageWidth(640)
  grabber.setImageHeight(480)
  grabber.setBitsPerPixel(CV_8U)
  grabber.setImageMode(ImageMode.COLOR)
  grabber.start()

  val greyImg = new BufferedImage(640, 480, BufferedImage.TYPE_BYTE_GRAY)

  var lastRecognitionTime = 0L
  val cvFont = new CvFont()
  cvFont.hscale(0.6f)
  cvFont.vscale(0.6f)
  cvFont.font_face(FONT_HERSHEY_SIMPLEX)

  val mat = new Mat(640, 480, CV_8UC3)
  val greyMat = new Mat(640, 480, CV_8U)
  while (true) {
    val img = grabber.grab()
    cvFlip(img, img, 1)
    mat.copyFrom(img.getBufferedImage)
    opencv_imgproc.cvtColor(mat, greyMat, opencv_imgproc.CV_BGR2GRAY, 1)
    canvas.showImage(greyMat)
  }

}
