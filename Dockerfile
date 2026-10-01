FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /srv

# Standard library only: no requirements installation needed, so the image
# builds offline and the runtime surface stays minimal.
COPY app ./app
COPY tests ./tests
COPY scripts ./scripts

RUN chmod +x scripts/verify.sh && python -m compileall -q app tests

EXPOSE 8000

HEALTHCHECK --interval=3s --timeout=3s --start-period=2s --retries=20 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/healthz', timeout=2).status==200 else 1)"

CMD ["python", "-m", "app.server"]
