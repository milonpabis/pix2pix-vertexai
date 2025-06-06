FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY trainer .

ENTRYPOINT ["python" "trainer/train.py"]
