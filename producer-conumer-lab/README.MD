Completed: 11/17/2018

This directory includes:

* `VideoDecoder.py`: which
  * defines a `ExtractFramesThread` class.
    * used to extract frames from video clip designated as `clipFileName`
    * extracted frames are placed into `extractBuffer` to be consumed
    
  * defines a `ConvertToGrayscaleThread` class.
	  * used to consume frames from `extractBuffer` and convert them to gray scale
	  * gray scale frames are placed into `grayBuffer` to be consumed
    
  * defines a `DisplayFramesThread` class.
		* used to consume frames from `grayBuffer` and display them in a window

## VideoDecoder.py

TO RUN: ./VideoDecoder.py

PRINCIPLE OF OPERATION:

The defined classes extend Thread and are called together.
Each thread communicates with the other by placeing/receiving items from two `Queue()`:

* `extractBuffer` is used to store frames extracted from `clipFileName`:
	* To communicate with `extractBuffer` there are two thread safe functions:
		* `extBufPut(image)` queues `image` into `extractBuffer`
		* `extBufGet()` pops next image from `extractBuffer`
      
* `grayBuffer` is used to store frames converted to grayscale
	* To communicate with `grayBuffer` there are two thread safe functions:
		* `grayBufPut(image)` queues `image` into `grayBuffer`
		* `grayBufGet()` pops next image from `grayBuffer`
Both `Queue()` are initialized to `BUF_SIZE`

Each buffer is thread safe because of two `Lock()`s:

* `exLock` is used to lock `extractBuffer`
* `grayLock` is used to lock `grayBuffer`
  
And by using four `Semaphore()`s:
* `extSemaEmpty` keeps count of empty cells in `extractBuffer` and is initialized to `BUF_SIZE`
* `extSemaFull` keeps count of full cells in `extractBuffer` and is initialized to `0`
* `graySemaEmpty` keeps count of empty cells in `grayBuffer` and is initialized to `BUF_SIZE`
* `graySemaFull` keeps count of full cells in `grayBuffer` and is initialized to `0`

The algorithm discussed in class was used to implement thread safety.
It was desgined using the following pseudocode:

```
Full = countingSem(0)
Empty = countingSem(N)
Qlock = mutex(1)

To put value v:
	Acquire rights to 1 empty q cell
		Empty.acquire()
	Insert into Q
		Qlock.acquire()
		Q.put(v)
		Qlock.release
	Release rights to 1 full queue cell
		Full.release()

To get a value v:
	Acquire rights to 1 full q cell
		Full.acquire()
	Delete from Q
		Qlock.acquire()
		V =Q.get()
		Qlock.release()
	Release rights to 1 empty q cell
		Empty.release()
```
