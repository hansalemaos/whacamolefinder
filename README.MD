# Get the difference between 2 images 

## Tested against Windows 10 / Python 3.11 / Anaconda

### pip install whacamolefinder

The WhacAMoleFinder class is designed for continuously comparing 
pairs of images and finding differences between them. 
It is used in scenarios where you want to monitor a changing image, 
such as a game or application interface, and detect any alterations or visual changes.

## The advantages:

### Continuous Image Comparison: 

The class allows you to continuously compare two consecutive screenshots or images, 
enabling real-time monitoring for changes.

### Difference Detection: 

It detects differences between the images using image processing techniques, 
highlighting regions where changes occur.

### Customizable Parameters: 

You can customize various parameters, such as the percentage of resizing, 
the color used for highlighting differences, the thickness 
of highlighting lines, and the threshold for difference detection. 
This flexibility allows you to adapt the class to different use cases and scenarios.

### Visualization: 

It provides an option to display the images with differences 
highlighted, making it easy to visualize what has changed.

### Interactive Control: 

The class includes a control mechanism to stop the comparison process, 
which is useful when you want to pause or terminate the monitoring.

### Yielding Results: 

It yields the results of the comparison as a list of Token namedtuples, 
providing information about the differences found in consecutive images.




```python

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
```