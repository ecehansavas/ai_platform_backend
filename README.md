# ESTRA
Easy Streaming Data Analysis Tool (ESTRA) is designed with the aim of creating an easy-to-use data stream analysis platform that serves the purpose of a quick and efficient tool to explore and prototype machine learning solutions on various datasets. ESTRA is developed as a web-based, scalable, extensible, and open-source data analysis tool with a user-friendly and easy to use user interface

ESTRA consist of 4 main components:
```
    User Interface -> Javascript / ReactJs (ai_platform_ui)
    Web Server -> Python / Django (ai_platform_backend)
    Database -> PostgreSQL
    Background worker -> Python (ai_platform_core)
```

ESTRA  provides  a  flexible  deployment  structure  depending  on  the  use  case.   For personal use, ESTRA can be run on a regular personal computer.  For a large scale use, every component can be deployed into their own servers and they can even be deployed as a load-balanced multi-instance fashion.

Details shared in the following link https://open.metu.edu.tr/handle/11511/89668 

## ai_platform_backend
Web server component is a lightweight web application written with Django which isa web application framework for Python.  Web server component is a kind of bridge between the client and the workers of ESTRA and actually contains minimal business logic. Web server provides endpoints to the user interface component for either taking new machine learning job requests or return the state of the machine learning jobs submitted  by  the  user,  including  the  states  and  results. Moreover,  the  web  server component  contains  the  database  models,  describing  the  table  and  index  structure used in the database.

## Running the code
```
heroku local
```