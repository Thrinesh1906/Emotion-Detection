# Test Texts for Emotion Detection Dashboard

Copy and paste these texts into the **"Live Prediction"** section of the dashboard (http://localhost:8501)

---

## 😊 JOY - Happy Expressions

### Strong Joy
1. **"I am so happy! This is the best day ever!"**
   - Expected: joy
   - Confidence: 70%+

2. **"That's amazing! I can't believe how wonderful this is!"**
   - Expected: joy
   - Confidence: 65%+

3. **"I love this! Thank you so much, you're the best!"**
   - Expected: joy
   - Confidence: 60%+

4. **"Fantastic news! I got the promotion I wanted!"**
   - Expected: joy
   - Confidence: 65%+

5. **"This is absolutely wonderful! I'm so excited!"**
   - Expected: joy
   - Confidence: 70%+

### Moderate Joy
6. **"That went pretty well, I'm pleased with the results."**
   - Expected: joy
   - Confidence: 50%+

7. **"Great! That sounds like a good plan."**
   - Expected: joy
   - Confidence: 45%+

---

## 😠 ANGER - Frustrated/Angry Expressions

### Strong Anger
1. **"I'm furious! This is completely unacceptable!"**
   - Expected: anger
   - Confidence: 70%+

2. **"I can't believe you did that! How dare you!"**
   - Expected: anger
   - Confidence: 65%+

3. **"Stop this right now! I hate this!"**
   - Expected: anger
   - Confidence: 75%+

4. **"This is absolutely ridiculous and infuriating!"**
   - Expected: anger
   - Confidence: 70%+

5. **"I'm so angry I could scream!"**
   - Expected: anger
   - Confidence: 65%+

### Moderate Anger
6. **"I'm frustrated with how this turned out."**
   - Expected: anger
   - Confidence: 50%+

7. **"This is really annoying and bothersome."**
   - Expected: anger
   - Confidence: 55%+

---

## 😢 SADNESS - Sad/Depressed Expressions

### Strong Sadness
1. **"I feel terrible and heartbroken about this."**
   - Expected: sadness
   - Confidence: 60%+

2. **"This is devastating news. I'm so sad."**
   - Expected: sadness
   - Confidence: 65%+

3. **"I miss them so much. Everything feels hopeless now."**
   - Expected: sadness
   - Confidence: 70%+

4. **"I can't stop crying. I'm completely broken."**
   - Expected: sadness
   - Confidence: 75%+

5. **"This is the worst day of my life."**
   - Expected: sadness
   - Confidence: 60%+

### Moderate Sadness
6. **"I'm not feeling great about this situation."**
   - Expected: sadness
   - Confidence: 45%+

7. **"This is unfortunate and disappointing."**
   - Expected: sadness
   - Confidence: 50%+

---

## 😨 FEAR - Scared/Anxious Expressions

### Strong Fear
1. **"I'm absolutely terrified right now!"**
   - Expected: fear
   - Confidence: 65%+

2. **"I'm scared! What if something goes wrong?"**
   - Expected: fear
   - Confidence: 60%+

3. **"This is horrifying. I'm panicking!"**
   - Expected: fear
   - Confidence: 70%+

4. **"I'm petrified of what might happen next."**
   - Expected: fear
   - Confidence: 65%+

5. **"I don't feel safe. I'm really worried."**
   - Expected: fear
   - Confidence: 55%+

### Moderate Fear
6. **"I'm a bit nervous about this situation."**
   - Expected: fear
   - Confidence: 45%+

7. **"I'm concerned and uncertain about the outcome."**
   - Expected: fear
   - Confidence: 40%+

---

## 😲 SURPRISE - Shocked/Amazed Expressions

### Strong Surprise
1. **"Oh my goodness! I didn't expect that at all!"**
   - Expected: surprise
   - Confidence: 65%+

2. **"Wow! That's shocking and unexpected!"**
   - Expected: surprise
   - Confidence: 60%+

3. **"Really? I can't believe this is happening!"**
   - Expected: surprise
   - Confidence: 55%+

4. **"That's incredible! I'm amazed!"**
   - Expected: surprise
   - Confidence: 60%+

5. **"Unbelievable! This is the last thing I expected!"**
   - Expected: surprise
   - Confidence: 65%+

### Moderate Surprise
6. **"Interesting! I didn't know that."**
   - Expected: surprise
   - Confidence: 40%+

7. **"Oh, that's surprising to hear."**
   - Expected: surprise
   - Confidence: 35%+

---

## 🤮 DISGUST - Revolting/Nasty Expressions

### Strong Disgust
1. **"That's absolutely disgusting! I can't stand it!"**
   - Expected: disgust
   - Confidence: 70%+

2. **"This is revolting and gross! Ugh!"**
   - Expected: disgust
   - Confidence: 65%+

3. **"That's vile and repulsive! I hate this!"**
   - Expected: disgust
   - Confidence: 70%+

4. **"How nauseating and awful this is!"**
   - Expected: disgust
   - Confidence: 65%+

### Moderate Disgust
5. **"That's pretty unpleasant and nasty."**
   - Expected: disgust
   - Confidence: 50%+

6. **"I really dislike this situation."**
   - Expected: disgust
   - Confidence: 40%+

---

## 😐 NEUTRAL - Calm/Rational Expressions

### Strong Neutral
1. **"Let me think about this calmly and rationally."**
   - Expected: neutral
   - Confidence: 95%+

2. **"Okay, let's discuss this in a structured way."**
   - Expected: neutral
   - Confidence: 90%+

3. **"Sure, that sounds reasonable."**
   - Expected: neutral
   - Confidence: 85%+

4. **"I understand the situation. Let me analyze it."**
   - Expected: neutral
   - Confidence: 90%+

5. **"Yes, I see what you mean. That makes sense."**
   - Expected: neutral
   - Confidence: 85%+

### Moderate Neutral
6. **"Maybe that could work. Let me consider it."**
   - Expected: neutral
   - Confidence: 70%+

7. **"Alright, I suppose that's an option."**
   - Expected: neutral
   - Confidence: 80%+

---

## 🔄 EMOTION TRANSITIONS - Multi-turn Conversations

### Transition 1: Neutral → Joy
**Previous emotion: neutral**
- Text: "Wait! That's wonderful news I just heard!"
- Expected: joy
- Shows how initial neutral state can shift to happiness

### Transition 2: Joy → Sadness
**Previous emotion: joy**
- Text: "Actually, I just realized something terrible about this."
- Expected: sadness
- Shows emotional shift from positive to negative

### Transition 3: Anger → Neutral
**Previous emotion: anger**
- Text: "Let me take a deep breath and think about this logically."
- Expected: neutral
- Shows de-escalation from anger to calm reasoning

### Transition 4: Fear → Joy
**Previous emotion: fear**
- Text: "Oh thank goodness! Everything turned out okay!"
- Expected: joy
- Shows relief and positive shift

### Transition 5: Sadness → Anger
**Previous emotion: sadness**
- Text: "I'm tired of this! I can't believe they did this to me!"
- Expected: anger
- Shows shift to frustration/anger from sadness

---

## 🎯 MIXED EMOTION - Tricky/Complex Expressions

1. **"I'm happy but also nervous about the future."**
   - Expected: mixed/joy (context dependent)
   - Confidence: 50-70%

2. **"I'm angry at the situation but sad about the outcome."**
   - Expected: anger or sadness (system will pick one)
   - Confidence: 50%+

3. **"That's both amazing and terrifying!"**
   - Expected: surprise or joy
   - Confidence: 50%+

4. **"I can't believe how disappointed I am!"**
   - Expected: sadness or disgust
   - Confidence: 55%+

5. **"This is frustrating but I'll handle it calmly."**
   - Expected: neutral (calm takes precedence)
   - Confidence: 60%+

---

## 📋 QUICK TEST PLAN

### For Testing (5 minutes)
1. "I'm so happy!" → Joy
2. "I'm furious!" → Anger
3. "I feel terrible" → Sadness
4. "I'm scared right now" → Fear
5. "Let me think calmly" → Neutral

### For Comprehensive Testing (15 minutes)
Run through all emotions above, one section at a time.

### For Production Validation (30 minutes)
Test all sections including transitions and mixed emotions.

---

## 🔍 What to Look For

✅ **Good Signs**
- Emotion matches the text content
- Confidence scores are reasonable (40-95%)
- Punctuation (!, ?) affects emotion detection
- Emotional words are recognized correctly

⚠️ **Issues to Watch For**
- Emotion doesn't match text → May need keyword list update
- Confidence always near 0.5 → Neural model needs retraining
- Punctuation ignored → Preprocessing issue
- Same emotion for very different texts → Model not discriminating

---

## 💡 Tips for Testing

1. **Start with strong emotions** - Use ! and clear emotional words
2. **Try edge cases** - Mixed emotions, sarcasm, negations
3. **Test transitions** - Set "Current Emotion State" in sidebar and test transitions
4. **Use Conversation Simulator** - Paste multi-line conversations for timeline view
5. **Check confidence scores** - Lower scores = more uncertain predictions

---

**Dashboard URL**: http://localhost:8501

Good luck testing! 🚀
