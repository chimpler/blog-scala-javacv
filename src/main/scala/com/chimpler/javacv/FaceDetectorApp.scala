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
object FaceDetectorApp extends App {

  case class Face(id: Int, faceRect: Rect, leftEyeRect: Rect, rightEyeRect: Rect)

  class FaceDetector() {
    val faceXml = FaceDetectorApp.getClass.getClassLoader.getResource("haarcascade_frontalface_alt.xml").getPath
    val faceCascade = new CascadeClassifier(faceXml)

    val leftEyeXml = FaceDetectorApp.getClass.getClassLoader.getResource("haarcascade_mcs_lefteye_alt.xml").getPath
    val leftEyeCascade = new CascadeClassifier(leftEyeXml)

    val rightEyeXml = FaceDetectorApp.getClass.getClassLoader.getResource("haarcascade_mcs_righteye_alt.xml").getPath
    val rightEyeCascade = new CascadeClassifier(rightEyeXml)

    val faceRects = new Rect(10)

    def detect(greyMat: Mat): mutable.Buffer[Face] = {
      val faces = mutable.Buffer.empty[Face]
      faceCascade.detectMultiScale(greyMat, faceRects)
      for(i <- 0 until faceRects.limit()) {
        val faceRect = faceRects.position(i)

        val leftFaceRect = new FaceRect()
        val leftFaceMat = new Mat(greyMat, )
        val faceMat = new Mat(greyMat, faceRect)

        val leftEyeRect = new Rect(1)
        val rightEyeRect = new Rect(1)
        leftEyeCascade.detectMultiScale(faceMat, leftEyeRect)
        rightEyeCascade.detectMultiScale(faceMat, rightEyeRect)

        faces += Face(i, faceRect, leftEyeRect, rightEyeRect)
      }
      faces
    }
  }

  val canvas = new CanvasFrame("Webcam")

  val faceDetector = new FaceDetector
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
  var faces = mutable.Buffer.empty[Face]
  while (true) {
    val img = grabber.grab()
    cvFlip(img, img, 1)

    if (System.currentTimeMillis() - lastRecognitionTime > 100) {
      mat.copyFrom(img.getBufferedImage)
      opencv_imgproc.cvtColor(mat, greyMat, opencv_imgproc.CV_BGR2GRAY, 1)
      opencv_imgproc.equalizeHist(greyMat, greyMat)
      faces = faceDetector.detect(greyMat)
      lastRecognitionTime = System.currentTimeMillis()
    }

    for(f <- faces) {
      cvRectangle(img,
        opencv_core.cvPoint(f.faceRect.x, f.faceRect.y),
        opencv_core.cvPoint(f.faceRect.x + f.faceRect.width, f.faceRect.y + f.faceRect.height),
        AbstractCvScalar.RED,
        1, CV_AA, 0)


      cvRectangle(img,
        opencv_core.cvPoint(f.faceRect.x + f.leftEyeRect.x, f.faceRect.y + f.leftEyeRect.y),
        opencv_core.cvPoint(f.faceRect.x + f.leftEyeRect.x + f.leftEyeRect.width, f.faceRect.y + f.leftEyeRect.y + f.leftEyeRect.height),
        AbstractCvScalar.BLUE,
        1, CV_AA, 0)

      cvRectangle(img,
        opencv_core.cvPoint(f.faceRect.x + f.rightEyeRect.x, f.faceRect.y + f.rightEyeRect.y),
        opencv_core.cvPoint(f.faceRect.x + f.rightEyeRect.x + f.rightEyeRect.width, f.faceRect.y + f.rightEyeRect.y + f.rightEyeRect.height),
        AbstractCvScalar.GREEN,
        1, CV_AA, 0)

      val cvPoint = opencv_core.cvPoint(f.faceRect.x, f.faceRect.y - 20)
      cvPutText(img, s"Face ${f.id}", cvPoint, cvFont, AbstractCvScalar.RED)
    }
    canvas.showImage(img)
  }

}
