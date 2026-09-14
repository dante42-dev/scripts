#!/usr/bin/env python3
import time
import os
import schedule
from generate import main

TARGET_TIME = os.getenv("BRIEF_SCHEDULE_TIME", "22:00")

print(f"[*] Scheduler service initialized. Starting initial compilation...")
try:
    main()
except Exception as e:
    print(f"[!] Initial run error: {e}")

print(f"[*] Scheduler running in background. Next execution set for {TARGET_TIME} daily.")
schedule.every().day.at(TARGET_TIME).do(main)

while True:
    schedule.run_pending()
    time.sleep(30)
