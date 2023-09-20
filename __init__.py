from collections import namedtuple

import cv2
from a_cv_imwrite_imread_plus import open_image_in_cv
import numexpr
import numpy as np
from a_cv2_easy_resize import add_easy_resize_to_cv2
from typing import Union, Any, Literal

Token = namedtuple(
    "Whac",
    [
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "center_x",
        "center_y",
        "width",
        "height",
        "area",
    ],
)

add_easy_resize_to_cv2()


def get_difference_of_2_pics(
    pic1: Any,
    pic2: Any,
    percent_resize: int = 10,
    draw_output: bool = False,
    draw_color: Union[tuple, list] = (255, 255, 0),
    thickness: int = 2,
    thresh: int = 3,
    maxval: int = 255,
    draw_on_1_or_2: Literal[1, 2] = 1,
) -> tuple:
    out = np.array([], dtype=np.uint16)
    first = open_image_in_cv(pic1, channels_in_output=2)
    first = cv2.easy_resize_image(
        first.copy(),
        width=None,
        height=None,
        percent=percent_resize,
        interpolation=cv2.INTER_AREA,
    )
    second = open_image_in_cv(pic2, channels_in_output=2)
    second = cv2.easy_resize_image(
        second.copy(),
        width=first.shape[1],
        height=first.shape[0],
        percent=None,
        interpolation=cv2.INTER_AREA,
    )
    gray = numexpr.evaluate(
        f"abs(first-second)",
        global_dict={},
        local_dict={"first": first, "second": second},
    ).astype(np.uint8)

    for i in range(0, 3):
        dilated = cv2.dilate(gray.copy(), None, iterations=i + 1)

    (T, thresh) = cv2.threshold(dilated, thresh, maxval, cv2.THRESH_BINARY)
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #
    if draw_output:
        if draw_on_1_or_2 == 1:
            out = open_image_in_cv(pic1, channels_in_output=3)
        else:
            out = open_image_in_cv(pic2, channels_in_output=3)
    allresults = []
    if len(cnts) == 2:
        cnts = cnts[0]
    elif len(cnts) == 3:
        cnts = cnts[1]
    for cc in cnts:
        box = cv2.boundingRect(cc)
        (x, y, w, h) = [int(x / (percent_resize / 100)) for x in box]
        allresults.append(
            Token(x, y, x + w, y + h, x + (w // 2), y + (h // 2), w, h, w * h)
        )
        if draw_output:
            cv2.rectangle(
                out, (x, y), (x + w, y + h), tuple(reversed(draw_color)), thickness
            )
            yva = y
            for key1, item1 in allresults[-1]._asdict().items():
                yva = yva + 25
                cv2.putText(
                    out,
                    f"{key1}: {item1}",
                    (x, yva),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    3,
                )
                cv2.putText(
                    out,
                    f"{key1}: {item1}",
                    (x, yva),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    draw_color,
                    1,
                )
    return allresults, out


class WhacAMoleFinder:
    r"""
    A class for continuously comparing pairs of images and finding differences between them.

    Args:
        screenshotiter: A generator or iterator that provides screenshots for comparison.

    Attributes:
        screenshotiter (generator): The iterator supplying screenshots.
        stop (bool): A flag to control the comparison process. Set to True to stop comparing.
        last_screenshot
        before_last_screenshot

    Methods:
        start_comparing: Begin the image comparison process and yield results.

    Example:
        from whacamolefinder import WhacAMoleFinder
        from fast_ctypes_screenshots import (
            ScreenshotOfOneMonitor,
        )

        # create your own screenshot function, you can use whatever you want (fast_ctypes_screenshots/adb/mss/pyautogui/...),
        # but it is important to use a loop and yield!
        def screenshot_iter_function():
            while True:
                with ScreenshotOfOneMonitor(
                    monitor=0, ascontiguousarray=False
                ) as screenshots_monitor:
                    yield screenshots_monitor.screenshot_one_monitor()



        # Create an instance of WhacAMoleFinder

        piit = WhacAMoleFinder(screenshot_iter_function) # pass the function without calling it!
        for ini, di in enumerate(
            piit.start_comparing(
                percent_resize=10,
                draw_output=True,
                draw_color=(255, 0, 255),
                thickness=20,
                thresh=3,
                maxval=255,
                draw_on_1_or_2=2,
                break_key="q",
            )
        ):
            print(di)
            if ini > 100:
            piit.stop = True
            print(piitlast_screenshot)
            print(piitbefore_last_screenshot)

    #You can compare 2 images without using the class:
    # Call the get_difference_of_2_pics function to compare the two images
    allresults, out = get_difference_of_2_pics(
        pic1='c:/pic1.jpg',
        pic2='c:/pic2.jpg',
        percent_resize=10,
        draw_output=True,  # Set to True to visualize the differences
        draw_color=(255, 0, 255),  # Custom color for highlighting differences
        thickness=2,  # Line thickness for highlighting
        thresh=3,  # Threshold for difference detection
        maxval=255,  # Maximum value for thresholding
        draw_on_1_or_2=1,  # Draw differences on the first image
    )
    """

    def __init__(self, screenshotiter):
        self.screenshotiter = screenshotiter
        self.stop = False
        self.last_screenshot = None
        self.before_last_screenshot = None

    def start_comparing(
        self,
        percent_resize: int = 10,
        draw_output: bool = False,
        draw_color: Union[tuple, list] = (255, 255, 0),
        thickness: int = 2,
        thresh: int = 3,
        maxval: int = 255,
        draw_on_1_or_2: Literal[1, 2] = 1,
        break_key: str = "q",
    ) -> list:
        r"""
        Start the image comparison process between consecutive screenshots and yield results.

        Args:
            percent_resize (int, optional): Percentage by which to resize images before comparison. Defaults to 10.
            draw_output (bool, optional): Whether to display images with differences highlighted. Defaults to False.
            draw_color (tuple or list, optional): Color used for highlighting differences. Defaults to (255, 255, 0).
            thickness (int, optional): Thickness of the highlighting lines. Defaults to 2.
            thresh (int, optional): Threshold for image difference detection. Defaults to 3.
            maxval (int, optional): Maximum value used in thresholding. Defaults to 255.
            draw_on_1_or_2 (int, optional): Whether to draw differences on the first or second image. Defaults to 1.
            break_key (str, optional): Key to press to stop the comparison process. Defaults to "q".

        Yields:
            list: A list of Token namedtuples representing differences found in consecutive images.


        """
        screenshotiter = self.screenshotiter()
        pic1 = next(screenshotiter)
        outpic = pic1.copy()
        while not self.stop:
            pic2 = next(screenshotiter)
            self.last_screenshot = pic2
            self.before_last_screenshot = pic1

            allresults, out = get_difference_of_2_pics(
                pic1=pic1.copy(),
                pic2=pic2.copy(),
                percent_resize=percent_resize,
                draw_output=draw_output,
                draw_color=draw_color,
                thickness=thickness,
                thresh=thresh,
                maxval=maxval,
                draw_on_1_or_2=draw_on_1_or_2,
            )
            if draw_output:
                if allresults:
                    outpic = out.copy()
                cv2.imshow("OUTPUT", outpic)

                # Wait for a key press and then close all windows
                if cv2.waitKey(25) & 0xFF == ord(break_key):
                    cv2.destroyAllWindows()
                    draw_output = False
            yield allresults
            pic1 = pic2.copy()
        if draw_output:
            cv2.destroyAllWindows()
