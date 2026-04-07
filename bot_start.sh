#!/bin/bash
# Запуск бота з PID-файлом — гарантує один екземпляр

PIDFILE="/tmp/capper_bot.pid"
LOGFILE="/tmp/capper_bot.log"
PROJECT="/Users/andrii.zubko/Desktop/projects/Capper"

# Вбиваємо попередній процес якщо є
if [ -f "$PIDFILE" ]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Зупиняємо старий процес (PID $OLD_PID)..."
        kill "$OLD_PID"
        sleep 2
    fi
    rm -f "$PIDFILE"
fi

# На всяк випадок — вбиваємо всі bot.main процеси
pkill -f "python.*bot.main" 2>/dev/null
sleep 1

# Запускаємо
cd "$PROJECT"
source venv/bin/activate
DATABASE_URL=postgresql://capper:capper@localhost:5432/capper \
    nohup python -m bot.main > "$LOGFILE" 2>&1 &

echo $! > "$PIDFILE"
echo "Бот запущений (PID $(cat $PIDFILE))"
echo "Логи: tail -f $LOGFILE"
