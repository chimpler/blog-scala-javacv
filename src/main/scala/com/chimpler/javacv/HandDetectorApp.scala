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
object HandDetectorApp extends App {

  case class Face(id: Int, faceRect: Rect, leftEyeRect: Rect, rightEyeRect: Rect)
  case class Hand(id: Int, handRect: Rect)

  class HandDetector() {
    val handXml = FaceDetectorApp.getClass.getClassLoader.getResource("haarcascade_hand.xml").getPath
    val handCascade = new CascadeClassifier(handXml)

    val handRects = new Rect(10)

    def detect(greyMat: Mat): mutable.Buffer[Hand] = {
      val hands = mutable.Buffer.empty[Hand]
      handCascade.detectMultiScale(greyMat, handRects)
      for(i <- 0 until handRects.limit()) {
        val handRect = handRects.position(i)
        hands += Hand(i, new Rect(handRect))
      }
      hands
    }
  }

  val canvas = new CanvasFrame("Webcam")

  val handDetector = new HandDetector
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
  var hands = mutable.Buffer.empty[Hand]
  while (true) {
    val img = grabber.grab()
    cvFlip(img, img, 1)

    if (System.currentTimeMillis() - lastRecognitionTime > 200) {
      mat.copyFrom(img.getBufferedImage)
      opencv_imgproc.cvtColor(mat, greyMat, opencv_imgproc.CV_BGR2GRAY, 1)
      opencv_imgproc.equalizeHist(greyMat, greyMat)
      hands = handDetector.detect(greyMat)
      lastRecognitionTime = System.currentTimeMillis()
    }

    for(h <- hands) {
      cvRectangle(img,
        opencv_core.cvPoint(h.handRect.x, h.handRect.y),
        opencv_core.cvPoint(h.handRect.x + h.handRect.width, h.handRect.y + h.handRect.height),
        AbstractCvScalar.YELLOW,
        1, CV_AA, 0)

      val cvPoint = opencv_core.cvPoint(h.handRect.x, h.handRect.y - 20)
      cvPutText(img, s"Hand ${h.id}", cvPoint, cvFont, AbstractCvScalar.YELLOW)
    }

    canvas.showImage(img)
  }

}
