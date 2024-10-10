# 图像的导入与显示
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
# image = cv.imread("E:/image.jpg",1)
#
# plt.imshow(image[:,:,::-1])
# plt.title("photo")
# plt.xticks = [0]
# plt.yticks = [0]
# plt.show()
# 画图操作
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
#
# image = np.zeros((512,512,3),np.uint8)
#
# cv.line(image,(0,0),(512,512),(255,255,255),5)
# cv.rectangle(image,(155,155),(355,355),(255,0,0),3)
# cv.circle(image,(256,256),100,(255,255,255),-1)
# cv.putText(image,"opencv",(256,256),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv.LINE_AA)
#
# plt.imshow(image[:,:,::-1])
# plt.title("photo")
# plt.xticks = [0]
# plt.yticks = [0]
# plt.show()
# 修改像素值
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
#
# image = np.zeros((512,512,3),np.uint8)
#
# image[100,100] = (255,0,0)
#
# plt.imshow(image[:,:,::-1])
# plt.title("photo")
# plt.xticks = [0]
# plt.yticks = [0]
# plt.show()
import cv2
# 获取图片的形状
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
#
# image = np.zeros((512,512,3),np.uint8)
#
# shape = image.shape
# print(shape)

# 获取图片的大小
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
#
# image = np.zeros((512,512,3),np.uint8)
#
# shape = image.size
# print(shape)

# 获取图片的数据类型
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
#
# image = np.zeros((512,512,3),np.uint8)
#
# shape = image.dtype
# print(shape)
# 图片的混合(用缩放来处理不同大小的照片)
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
#
# image = cv.imread("E:/02.jpg")
# lin = cv.imread("E:/lin.jpg")
# lin2 = cv.resize(lin,None,fx = 1,fy = 2284/1920,interpolation=cv2.INTER_LINEAR)
# image1 = cv.addWeighted(image,0.7,lin2,0.3,0)     # 图像混合函数，入口参数为权重，相加为1
# plt.imshow(image1[:,:,::-1])
# plt.show()
# 图像的缩放+对大小不同的图片进行相加
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
# image = cv.imread("E:/02.jpg")
# shape1 = image.shape
# lin = cv.imread("E:/lin.jpg")
# shape2 = lin.shape
# print(shape1,shape2)
# lin2 = cv.resize(lin,None,fx = 1,fy = 2284/1920,interpolation=cv2.INTER_LINEAR)
# shape4 = lin2.shape
# print(shape4)
# image1 = cv.add(image,lin2)
# plt.imshow(image1[:,:,::-1])
# plt.show()
# 图象的移动
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
# lin = cv.imread("E:/lin.jpg")
# y,x = lin.shape[:2]
# print(x,y)
# M = np.float32([[1,0,100],[0,1,50]])
# image1 = cv.warpAffine(lin,M,(x,y))
# plt.imshow(image1[:,:,::-1])
# plt.show()
# 图像的旋转
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
# lin = cv.imread("E:/lin.jpg")
# y,x = lin.shape[:2]
# M = cv2.getRotationMatrix2D((x/2,y/2),90,1)
# image1 = cv.warpAffine(lin,M,(x,y))
# plt.imshow(image1[:,:,::-1])
# plt.show()
# 图像的仿射变换
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
# lin = cv.imread("E:/lin.jpg")
# y,x = lin.shape[:2]
# pts1 = np.float32([[50,50],[200,50],[50,200]])
# pts2 = np.float32([[100,100],[200,50],[100,250]])
# M = cv2.getAffineTransform(pts1,pts2)
# image1 = cv.warpAffine(lin,M,(x,y))
# plt.imshow(image1[:,:,::-1])
# plt.show()
# 图像的透射变化
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
# lin = cv.imread("E:/lin.jpg")
# y,x = lin.shape[:2]
# pts1 = np.float32([[50,50],[200,50],[50,200],[250,300]])
# pts2 = np.float32([[100,100],[200,50],[100,250],[280,350]])
# M = cv2.getPerspectiveTransform(pts1,pts2)
# image1 = cv.warpAffine(lin,M,(x,y))
# plt.imshow(image1[:,:,::-1])
# plt.show()
# 图像金字塔，对图层进行采样
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
# lin = cv.imread("E:/lin.jpg")
# lin_up = cv.pyrUp(lin)
# lin_down = cv.pyrDown(lin)
# cv.imshow("1",lin)
# cv.imshow("2",lin_up)
# cv.imshow("3",lin_down)
# cv.waitKey(0)
# cv.destroyWindow()



