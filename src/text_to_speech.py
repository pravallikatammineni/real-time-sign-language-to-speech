"""
Text-to-Speech Module
Provides utilities for converting text to speech
"""

import pyttsx3


class TextToSpeech:
    """
    Converts text to speech using pyttsx3 engine.
    """

    def __init__(self, rate=150):
        """
        Initialize the TTS engine.
        
        Args:
            rate: Speech rate (words per minute)
        """
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.last_spoken_text = ""

    def speak(self, text):
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to be spoken
        """
        if text and text != self.last_spoken_text:
            self.engine.say(text)
            self.engine.runAndWait()
            self.last_spoken_text = text

    def speak_forced(self, text):
        """
        Convert text to speech and play it (even if repeated).
        
        Args:
            text: Text to be spoken
        """
        if text:
            self.engine.say(text)
            self.engine.runAndWait()
            self.last_spoken_text = text

    def set_rate(self, rate):
        """
        Set the speech rate.
        
        Args:
            rate: Words per minute
        """
        self.engine.setProperty('rate', rate)

    def set_volume(self, volume):
        """
        Set the volume (0.0 to 1.0).
        
        Args:
            volume: Volume level
        """
        self.engine.setProperty('volume', volume)

    def get_voices(self):
        """
        Get available voices.
        
        Returns:
            list: Available voice objects
        """
        return self.engine.getProperty('voices')

    def set_voice(self, voice_id):
        """
        Set the voice.
        
        Args:
            voice_id: Voice ID from available voices
        """
        self.engine.setProperty('voice', voice_id)

    def close(self):
        """Close the TTS engine."""
        self.engine.stop()
