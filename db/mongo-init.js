db = db.getSiblingDB('interview_buddy');

db.createCollection('frames');
db.frames.createIndex({ sessionId: 1, t: 1 });

db.createCollection('windows');
db.windows.createIndex({ sessionId: 1, tStart: 1 });

db.createCollection('realtime_events');
db.realtime_events.createIndex({ sessionId: 1, t: 1 });
db.realtime_events.createIndex({ createdAt: 1 }, { expireAfterSeconds: 86400 });
