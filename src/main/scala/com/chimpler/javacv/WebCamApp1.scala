package com.chimpler.javacv

import org.bytedeco.javacpp.opencv_core._
import org.bytedeco.javacv.FrameGrabber.ImageMode
import org.bytedeco.javacv.{OpenCVFrameGrabber, CanvasFrame}

/**
 * Created by frederic on 7/13/14.
 */
object WebCamApp1 extends App {
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

  val cvFont = new CvFont()
  cvFont.hscale(0.6f)
  cvFont.vscale(0.6f)
  cvFont.font_face(FONT_HERSHEY_SIMPLEX)

  while (true) {
    val img = grabber.grab()
    cvFlip(img, img, 1)

    canvas.showImage(img)
  }

}
