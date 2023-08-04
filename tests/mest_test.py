import time

import cv2
import numpy as np

from my_insightface.insightface.utils.my_tools import detect_cameras


def get_FOURCC(*resolution):
    cap = cv2.VideoCapture(0)

    #  设置帧数
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(resolution[0]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(resolution[1]))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    get_fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
    print(f"after setting, The video  codec  is {codec}")
    frames = 0
    start_time = time.time()
    try:
        while True:
            if_true, frame = cap.read()
            if if_true:
                cv2.imshow('test', frame)
                frames += 1
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(e)
    finally:
        # 释放VideoCapture对象
        end_time = time.time()
        print("resolution is ", resolution)
        print(f"measured FPS of the video is {frames / (end_time - start_time)}")
        print(f"get FPS of the video is {get_fps}")
        cap.release()
import re

def replace_emoji_with_entity(html):
    return re.sub(r'([^\x00-\x7F]+)',
                  lambda c: ''.join('&#{};'.format(ord(char)) for char in c.group(1)),
                  html)

def main():
    html = """
    <div class="row">
<div class="col-lg-12">
    <div class="justify-content-between d-flex align-items-center mt-3 mb-4">
        <h5 class="mb-0 pb-1 text-decoration-underline"><br></h5>
    </div>
</div>

<div class="col-12">
    <table class="body-wrap" style="font-family: Roboto, sans-serif; box-sizing: border-box; font-size: 14px; width: 100%; background-color: transparent; margin: 0px; --darkreader-inline-bgcolor: transparent;" data-darkreader-inline-bgcolor="">
        <tbody><tr style="font-family: 'Roboto', sans-serif; box-sizing: border-box; font-size: 14px; margin: 0;">
            <td style="font-family: 'Roboto', sans-serif; box-sizing: border-box; font-size: 14px; vertical-align: top; margin: 0;" valign="top"></td>
            <td class="container" width="600" style="font-family: 'Roboto', sans-serif; box-sizing: border-box; font-size: 14px; vertical-align: top; display: block !important; max-width: 600px !important; clear: both !important; margin: 0 auto;" valign="top">
                <div class="content" style="font-family: 'Roboto', sans-serif; box-sizing: border-box; font-size: 14px; max-width: 600px; display: block; margin: 0 auto; padding: 20px;">
                    <table class="main" width="100%" cellpadding="0" cellspacing="0" itemprop="action" itemscope="" itemtype="http://schema.org/ConfirmAction" style="font-family: Roboto, sans-serif; box-sizing: border-box; font-size: 14px; border-radius: 3px; margin: 0px; border: none; --darkreader-inline-border-top: initial; --darkreader-inline-border-right: initial; --darkreader-inline-border-bottom: initial; --darkreader-inline-border-left: initial;" data-darkreader-inline-border-top="" data-darkreader-inline-border-right="" data-darkreader-inline-border-bottom="" data-darkreader-inline-border-left="">
                        <tbody><tr style="font-family: 'Roboto', sans-serif; font-size: 14px; margin: 0;">
                            <td class="content-wrap" style="font-family: Roboto, sans-serif; box-sizing: border-box; color: rgb(73, 80, 87); font-size: 14px; vertical-align: top; margin: 0px; padding: 30px; box-shadow: rgba(30, 32, 37, 0.06) 0px 3px 15px; border-radius: 7px; background-color: rgb(255, 255, 255); --darkreader-inline-color: #b5a995; --darkreader-inline-boxshadow: rgba(57, 55, 50, 0.06) 0px 3px 15px; --darkreader-inline-bgcolor: #383631;" valign="top" data-darkreader-inline-color="" data-darkreader-inline-boxshadow="" data-darkreader-inline-bgcolor="">
                                
                                <table width="100%" cellpadding="0" cellspacing="0" style="font-family: 'Roboto', sans-serif; box-sizing: border-box; font-size: 14px; margin: 0;">
                                    <tbody><tr style="font-family: 'Roboto', sans-serif; box-sizing: border-box; font-size: 14px; margin: 0;">
                                        <td class="content-block" style="font-family: 'Roboto', sans-serif; box-sizing: border-box; font-size: 14px; vertical-align: top; margin: 0; padding: 0 0 20px;" valign="top">
                                            <div style="text-align: center;margin-bottom: 15px;">
                                                <img src="/cgi-bin/viewfile?f=18860D25BA5D8C735F6B0E83A3A1B37FB32F8CAEED732764066DCA62FB7BA89F6667FC0F0483B0B4572B7C1D153C37F476BCB38C33E5363BA3ABF0F9FFD87289B1C756D8B13417FA1173F45B9CBC413486B014A5869B5E78C82AA75711592423&amp;mailid=ZL1704-SsO1JbRWzuAsCWQAwEQQbd8&amp;sid=wUgVpDXhPBLebAFl&amp;net=3381210142">
                                            </div>
                                        </td>
                                    </tr>
                                    <tr style="font-family: 'Roboto', sans-serif; box-sizing: border-box; font-size: 14px; margin: 0;">
                                        <td class="content-block" style="font-family: 'Roboto', sans-serif; box-sizing: border-box; font-size: 24px; vertical-align: top; margin: 0; padding: 0 0 10px;  text-align: center;" valign="top">
                                            <h5 style="font-family: 'Roboto', sans-serif; font-weight: 500;">Hey ~ 😎</h5><h5 style="font-family: 'Roboto', sans-serif; font-weight: 500;">Let's Verify your email ! 💌</h5>
                                        </td>
                                    </tr>
                                    <tr style="font-family: 'Roboto', sans-serif; box-sizing: border-box; font-size: 14px; margin: 0;">
                                        <td class="content-block" style="font-family: Roboto, sans-serif; color: rgb(135, 138, 153); box-sizing: border-box; font-size: 15px; vertical-align: top; margin: 0px; padding: 0px 0px 26px; text-align: center; --darkreader-inline-color: #a59985;" valign="top" data-darkreader-inline-color="">🙆🏻‍♀️Yes, we know its🙆🏻‍♂️<br><p style="margin-bottom: 13px;">👋🏻 An email to verify your account.👏🏻</p>
                                            <p style="margin-bottom: 0;">&nbsp;Press&nbsp; 👇🏻👇🏻👇🏻 to&nbsp; validate your email address&nbsp; &nbsp; &nbsp;</p><p style="margin-bottom: 0;">&nbsp;To get started your fantastic journey 🎉</p>
                                        </td>
                                    </tr>
                                    <tr style="font-family: 'Roboto', sans-serif; box-sizing: border-box; font-size: 14px; margin: 0;">
                                        <td class="content-block" itemprop="handler" itemscope="" itemtype="http://schema.org/HttpActionHandler" style="font-family: 'Roboto', sans-serif; box-sizing: border-box; font-size: 14px; vertical-align: top; margin: 0; padding: 0 0 22px; text-align: center;" valign="top">
                                            <a itemprop="url" style="font-family: Roboto, sans-serif; box-sizing: border-box; font-size: 0.8125rem; color: rgb(255, 255, 255); text-decoration: none; font-weight: 400; text-align: center; cursor: pointer; display: inline-block; border-radius: 0.25rem; text-transform: capitalize; background-color: rgb(64, 81, 137); margin: 0px; border-color: rgb(64, 81, 137); border-style: solid; border-width: 1px; padding: 0.5rem 0.9rem; --darkreader-inline-color: #e0d4be; --darkreader-inline-bgcolor: #545560; --darkreader-inline-border-top: #5e606e; --darkreader-inline-border-right: #5e606e; --darkreader-inline-border-bottom: #5e606e; --darkreader-inline-border-left: #5e606e;" data-darkreader-inline-color="" data-darkreader-inline-bgcolor="" data-darkreader-inline-border-top="" data-darkreader-inline-border-right="" data-darkreader-inline-border-bottom="" data-darkreader-inline-border-left="">Verify Your Email</a>
                                        </td>
                                    </tr>
                                    <tr style="font-family: 'Roboto', sans-serif; box-sizing: border-box; font-size: 14px; margin: 0;">
                                        <td class="content-block" style="color: rgb(135, 138, 153); text-align: center; font-family: Roboto, sans-serif; box-sizing: border-box; font-size: 14px; vertical-align: top; margin: 0px; padding: 5px 0px 0px; --darkreader-inline-color: #a59985;" valign="top" data-darkreader-inline-color="">
                                            <p style="margin-bottom: 10px;">Or verify using this link: </p>
                                            <a href="https://themesbrand.com/velzon/" target="_blank">https://themesbrand.com/velzon/</a>
                                        </td>
                                    </tr>
                                </tbody></table>
                            </td>
                        </tr>
                    </tbody></table>
                    <div style="text-align: center; margin: 25px auto 0px auto;font-family: 'Roboto', sans-serif;">
                        <h4 style="font-weight: 500; line-height: 1.5;font-family: 'Roboto', sans-serif;">Need Help ?</h4>
                        <p style="color: rgb(135, 138, 153); line-height: 1.5; --darkreader-inline-color: #a59985;" data-darkreader-inline-color="">Please send and feedback or bug info to <a href="mailto:zhouge1831@gmail.com" style="font-weight: 500;">zhouge1831@gmail.com</a></p>
                        <p style="font-family: Roboto, sans-serif; font-size: 14px; color: rgb(152, 166, 173); margin: 0px; --darkreader-inline-color: #afa38e;" data-darkreader-inline-color="">2022 Velzon. Design &amp; Develop by Themesbrand</p>
                    </div>
                </div>
            </td>
        </tr>
    </tbody></table>
    
</div>

</div>"""
    new_html = replace_emoji_with_entity(html)
    print(new_html,file=open('new.html','w',encoding='utf-8'))
if __name__ == '__main__':
    main()
