# RedisEdge Realtime Video Analytics

This is an example of using Redis Streams, RedisGears, RedisAI and RedisTimeSeries for Realtime Video Analytics (i.e. counting people).

TBD: animated gif

## Overview

This project demonstrates a possible deployment of the RedisEdge stack that provides realtime analytics of video streams.

The following diagram depicts the system's parts.

![Overview](/overview.png)

1. A video stream producer adds a captured frame to a Redis Stream.
2. The new frame triggers the execution of a RedisGear that:
    1. Downsamples the frame rate of the input stream, if needed.
    2. Resizes the input frame to the model's requirements.
    3. Calls RedisAI to execute an object recognition model on the frame.
    4. Stores the model's outputs (i.e. people counted and their whereabouts inside the frame) in Redis Stream and TimeSeriess.
3. A video web server renders the final image based on realtime data from Redis' Streams.
4. Time series are exported from Redis to Prometheus, enabling visualization with Grafana's dashboards.

### The RedisEdge Stack

The RedisEdge stack consists of a the latest Redis stable release and select RedisLabs modules inteded to be used in Edge computing. For more information refer to [RedisEdge](https://github.com/RedisLabs/redis-edge-docker).

### YOLO

You Look Only Once, or YOLO for shorts ([good overview](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/single-shot-detectors/yolo.html)), is an object detection neural network. This project uses the "tiny" YOLOv3 model.

## How to get it

Prerequisites:
- [git](https://git-scm.com/)
- [git-lfs](https://git-lfs.github.com/)

```bash
$ git clone https://github.com/RedisGears/EdgeRealtimeVideoAnalytics.git
$ cd EdgeRealtimeVideoAnalytics
$ git lfs install; git lfs checkout
```

## How to run it locally

### The RedisEdge stack

Refer to the build/installation instructions of the following projects to set up a Redis server with the relevant Redis modules. This application's connections default to `redis://localhost:6379`.

* [Redis](https://redis.io)
* [RedisGears](https://oss.redislabs.com/redisgears/)
* [RedisTimeSeries](https://oss.redislabs.com/redistimeseries/)
* [RedisAI](https://oss.redislabs.com/redisai/)

Note that you'll also need to install the Pythonic [`requirements.txt`](/redisedge/requirements.txt) for the embedded RedisGears Python interpreter.

### (optional) Prometheus and Grafana

Refer to the build/installation instructions of the following projects to set up Prometheus, Grafana and the RedisTimeSeries adapter:

* Prometheus: [Installation](https://prometheus.io/), [config sample](/prometheus/config.yml)
* Grafana: [Installation](https://grafana.com/), [config sample](/grafana/config.ini), [datasource sample](/grafana/provisioning/datasources/prometheus.yaml), [dashboard samples](/grafana/dashboards/)
* [prometheus-redistimeseries-adapter](https://github.com/RedisTimeSeries/prometheus-redistimeseries-adapter)

## The application

The application is implemented in Python 3, and consists of the following parts:

- [`init.py`](/app/init.py): this initializes Redis with the RedisAI model, RedisTimeSeries downsampling rules and the RedisGears gear.
- [`capture.py`](/app/capture.py): captures video stream frames from a webcam or image/video file and stores it in a Redis Stream.
- [`server.py`](/app/server.py): a web server that serves a rendered image composed of the raw frame and the model's detections.
- [`top.py`](/app/top.py): prints runtime performance metrics. Optional (i.e. to be run manually).
- [`gear.py`](/app/gear.py): the Redis gear that glues the pipeline.

To run the application you'll need Python v3.6 or higher. Install the application's library dependencies with the following - it is recommended that you use `virtualenv` or similar:

```sh
$ virtualenv -p python3.6 venv
$ source venv/bin/activate
$ cd app
$ pip install -r requirements.txt
```

The application's parts are set up with default values that are intended to allow it to run "out of the box". For example, to run the capture process you only need to type:

```sh
$ python capture.py
```

This will run the capture process from device id 0.

However. Most default values can be overridden from command line - invoke the application's parts with the `--help` switch to learn of these.

## How to run it with Docker Compose

Prerequisites:
- Docker

### Ubuntu 18.04 (and possibly older versions, other Debian-based distros and all Linux in general)

#### Fully-dockerized

The following will spin up a fully dockerized environment:

```sh
$ docker-compose up
```

Alternatively, you can bring up a lean environment (no fancy [UI](#UI)) with:
```sh
$ docker-compose up redisedge init capture server
```

#### Partially-dockerize

For performance gains, a [local Docker composition](/docker-compose.local.yaml) that includes only the app's initializer, server, grafana, prometheus and the RedisTimeSeries adapter is provided. Put differently, you need to provide the RedisEdge server and a video stream.

To use it, first make sure that you start a your RedisEdge server, e.g.:

```sh
$ ./redisedge/run.sh
```

Then, you can run the rest of the stack with:
```sh
$ docker-compose -f docker-compose.local.yaml up
```

Finally, make sure you actually start capturing something locally, e.g.:
```sh
$ python app/capture.py app/data/countvoncount.mp4
```

**Note:** when switching between fully- and partially-dockerized runs, make sure you rebuild (e.g. `docker-compose up --build`). 

### macOS

#### Fully-dockerized

According to current wisdom, it is impossible to use the webcam from a Docker container on macOS. To work around that, always run the `capture.py` locally on the host.

#### Partially-dockerized (a.k.a. local)

According to current wisdom, 'host' mode networking is a myth on macOS. Hence, the partially-dockerized mode is not available. TL;DR - it is either (almost) fully-dockerized or local for you.

## Performance notes

1. Docker makes everything slower.
2. The bigger your captured video, the slower this would run.
3. If you don't have a GPU but your CPU supports additional optimizations, consider compiling TensorFlow with them (and then RedisAI) and running RedisEdge locally.
4. If you have a GPU, use it (i.e. compile RedisAI for it and `python init.py --device GPU`)

### Unofficial metrics

Metrics sampled by capturing the [Count's video](/app/data/countvoncount.mp4) using the [application's top tool](/app/top.py).


#### MacBook Pro TBD

* macOS, dockerized: TBD msec
* Ubuntu 18.04, dockerized/local: out: ~4.5 fps, ~190 msec
* Ubuntu 18.04, local with CPU optimizations - out: ~8 fps, total: ~110 msec

## UI

The application's raw video ([server.py](/app/server.py)) should be at http://localhost:5000/video.

The Docker Compose setup alos comes with a pre-provisioned Grafana server that should be at http://localhost:3000/ (admin/admin). It is configured with the data source and dashboard, so once you log in:

1. Choose 'Dashboards' from the left-dock (2nd icon)
2. Click 'Manage'
3. Click 'Video Stream Analytics'
4. Voila!

## Known issues, limitations and todos

* TBD
