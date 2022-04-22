import datetime as dt
from datetime import date
from datetime import timedelta

start = date.today() - timedelta(days=730)
end = date.today() - timedelta(days=1)

print(start)
print(end)
