class AudioSynchronizer:
    def apply_audio_effects(self, clip):
        """
        Applies audio normalization and fade in/out to the clip.
        Returns the modified clip.
        """
        try:
            if clip.audio is None:
                return clip

            clip = self.normalize_audio(clip)

            fade_duration = min(0.5, clip.duration / 4)

            try:
                from moviepy.audio.fx.audio_fadein import audio_fadein
                from moviepy.audio.fx.audio_fadeout import audio_fadeout
                clip = clip.fx(audio_fadein, fade_duration)
                clip = clip.fx(audio_fadeout, fade_duration)
            except Exception:
                pass

            return clip
        except Exception:
            return clip

    def normalize_audio(self, clip):
        """
        Normalizes audio volume to a target level.
        Returns the modified clip.
        """
        try:
            if clip.audio is None:
                return clip

            target_volume = 0.8
            clip = clip.volumex(target_volume)
            return clip
        except Exception:
            return clip
