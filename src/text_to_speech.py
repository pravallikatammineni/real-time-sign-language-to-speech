"""
Text-to-Speech Module
Speaks text out loud using the system's built-in TTS engine.
Works offline, no fancy cloud API needed.
"""

import pyttsx3


class TextToSpeech:
    """
    Converts text to speech using pyttsx3 (offline TTS engine).
    
    Wraps pyttsx3 to give you a simpler interface and prevents
    saying the same thing over and over again.
    """

    def __init__(self, rate=150):
        """
        Set up the TTS engine.
        
        Args:
            rate: How fast to talk (words per minute).
                  150 is normal speed, 200 is fast, 100 is slow.
                  Experiment to find what sounds natural to you.
        """
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.last_spoken_text = ""

    def speak(self, text):
        """
        Say the text out loud (if it's different from last time).
        
        This is smart: if you ask it to speak "A" ten times in a row,
        it only speaks once. Good for preventing annoying repetition.
        
        Args:
            text: What to say
        """
        if text and text != self.last_spoken_text:
            self.engine.say(text)
            self.engine.runAndWait()
            self.last_spoken_text = text

    def speak_forced(self, text):
        """
        Say it regardless of what was just said.
        
        Use this if you want to speak the same text multiple times in a row.
        
        Args:
            text: What to say
        """
        if text:
            self.engine.say(text)
            self.engine.runAndWait()
            self.last_spoken_text = text

    def set_rate(self, rate):
        """
        Change how fast the speech is.
        
        Args:
            rate: Words per minute (150 = normal)
        """
        self.engine.setProperty('rate', rate)

    def set_volume(self, volume):
        """
        Make it louder or quieter.
        
        Args:
            volume: 0.0 (silent) to 1.0 (maximum volume)
        """
        self.engine.setProperty('volume', volume)

    def get_voices(self):
        """
        See what voices are available on this system.
        
        Different systems have different voices. Might have male/female options.
        
        Returns:
            list: Voice objects you can use with set_voice()
        """
        return self.engine.getProperty('voices')

    def set_voice(self, voice_id):
        """
        Pick a different voice.
        
        Get available voices from get_voices() first.
        
        Args:
            voice_id: ID of the voice to use
        """
        self.engine.setProperty('voice', voice_id)

    def close(self):
        """Close the TTS engine."""
        self.engine.stop()
