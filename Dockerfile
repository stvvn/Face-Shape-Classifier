FROM continuumio/anaconda3:latest
COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/
RUN pip install -r requirements.txt
CMD python face_shape_server.py