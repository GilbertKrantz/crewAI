"""Tests for OpenAI Responses API support in CrewAI."""

from unittest.mock import MagicMock, patch

import pytest

from crewai.llm import LLM


class TestResponsesAPISupport:
    """Test suite for OpenAI Responses API parameters support."""

    def test_llm_accepts_modalities_parameter(self) -> None:
        """Test that LLM class accepts modalities parameter."""
        llm = LLM(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            is_litellm=True,
        )
        assert llm.modalities == ["text", "audio"]

    def test_llm_accepts_audio_parameter(self) -> None:
        """Test that LLM class accepts audio parameter."""
        audio_config = {"voice": "alloy", "format": "wav"}
        llm = LLM(
            model="gpt-4o-audio-preview",
            audio=audio_config,
            is_litellm=True,
        )
        assert llm.audio == audio_config

    def test_llm_accepts_prediction_parameter(self) -> None:
        """Test that LLM class accepts prediction parameter."""
        prediction_config = {"type": "content", "content": "test"}
        llm = LLM(
            model="gpt-4o",
            prediction=prediction_config,
            is_litellm=True,
        )
        assert llm.prediction == prediction_config

    def test_llm_passes_modalities_to_litellm(self) -> None:
        """Test that modalities parameter is passed to litellm.completion."""
        llm = LLM(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            is_litellm=True,
        )

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Test response",
                            tool_calls=None,
                        )
                    )
                ]
            )

            llm.call("Test message")

            # Verify that litellm.completion was called with modalities
            call_args = mock_completion.call_args
            assert call_args is not None
            assert "modalities" in call_args.kwargs
            assert call_args.kwargs["modalities"] == ["text", "audio"]

    def test_llm_passes_audio_to_litellm(self) -> None:
        """Test that audio parameter is passed to litellm.completion."""
        audio_config = {"voice": "alloy", "format": "wav"}
        llm = LLM(
            model="gpt-4o-audio-preview",
            audio=audio_config,
            is_litellm=True,
        )

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Test response",
                            tool_calls=None,
                        )
                    )
                ]
            )

            llm.call("Test message")

            # Verify that litellm.completion was called with audio
            call_args = mock_completion.call_args
            assert call_args is not None
            assert "audio" in call_args.kwargs
            assert call_args.kwargs["audio"] == audio_config

    def test_llm_passes_prediction_to_litellm(self) -> None:
        """Test that prediction parameter is passed to litellm.completion."""
        prediction_config = {"type": "content", "content": "test"}
        llm = LLM(
            model="gpt-4o",
            prediction=prediction_config,
            is_litellm=True,
        )

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Test response",
                            tool_calls=None,
                        )
                    )
                ]
            )

            llm.call("Test message")

            # Verify that litellm.completion was called with prediction
            call_args = mock_completion.call_args
            assert call_args is not None
            assert "prediction" in call_args.kwargs
            assert call_args.kwargs["prediction"] == prediction_config

    def test_llm_passes_all_responses_api_params_to_litellm(self) -> None:
        """Test that all Responses API parameters are passed to litellm.completion."""
        audio_config = {"voice": "alloy", "format": "wav"}
        prediction_config = {"type": "content", "content": "test"}

        llm = LLM(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio=audio_config,
            prediction=prediction_config,
            is_litellm=True,
        )

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Test response",
                            tool_calls=None,
                        )
                    )
                ]
            )

            llm.call("Test message")

            # Verify that litellm.completion was called with all parameters
            call_args = mock_completion.call_args
            assert call_args is not None
            assert "modalities" in call_args.kwargs
            assert call_args.kwargs["modalities"] == ["text", "audio"]
            assert "audio" in call_args.kwargs
            assert call_args.kwargs["audio"] == audio_config
            assert "prediction" in call_args.kwargs
            assert call_args.kwargs["prediction"] == prediction_config

    def test_openai_native_accepts_modalities_parameter(self) -> None:
        """Test that OpenAI native provider accepts modalities parameter."""
        from crewai.llms.providers.openai.completion import OpenAICompletion

        llm = OpenAICompletion(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            api_key="test-key",
        )
        assert llm.modalities == ["text", "audio"]

    def test_openai_native_accepts_audio_parameter(self) -> None:
        """Test that OpenAI native provider accepts audio parameter."""
        from crewai.llms.providers.openai.completion import OpenAICompletion

        audio_config = {"voice": "alloy", "format": "wav"}
        llm = OpenAICompletion(
            model="gpt-4o-audio-preview",
            audio=audio_config,
            api_key="test-key",
        )
        assert llm.audio == audio_config

    def test_openai_native_accepts_prediction_parameter(self) -> None:
        """Test that OpenAI native provider accepts prediction parameter."""
        from crewai.llms.providers.openai.completion import OpenAICompletion

        prediction_config = {"type": "content", "content": "test"}
        llm = OpenAICompletion(
            model="gpt-4o",
            prediction=prediction_config,
            api_key="test-key",
        )
        assert llm.prediction == prediction_config

    def test_openai_native_passes_modalities_to_api(self) -> None:
        """Test that OpenAI native provider passes modalities to API."""
        from crewai.llms.providers.openai.completion import OpenAICompletion

        llm = OpenAICompletion(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            api_key="test-key",
        )

        with patch.object(llm.client.chat.completions, "create") as mock_create:
            mock_create.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Test response",
                            tool_calls=None,
                        )
                    )
                ],
                usage=MagicMock(
                    prompt_tokens=10, completion_tokens=20, total_tokens=30
                ),
            )

            llm.call("Test message")

            # Verify that OpenAI API was called with modalities
            call_args = mock_create.call_args
            assert call_args is not None
            assert "modalities" in call_args.kwargs
            assert call_args.kwargs["modalities"] == ["text", "audio"]

    def test_openai_native_passes_audio_to_api(self) -> None:
        """Test that OpenAI native provider passes audio to API."""
        from crewai.llms.providers.openai.completion import OpenAICompletion

        audio_config = {"voice": "alloy", "format": "wav"}
        llm = OpenAICompletion(
            model="gpt-4o-audio-preview",
            audio=audio_config,
            api_key="test-key",
        )

        with patch.object(llm.client.chat.completions, "create") as mock_create:
            mock_create.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Test response",
                            tool_calls=None,
                        )
                    )
                ],
                usage=MagicMock(
                    prompt_tokens=10, completion_tokens=20, total_tokens=30
                ),
            )

            llm.call("Test message")

            # Verify that OpenAI API was called with audio
            call_args = mock_create.call_args
            assert call_args is not None
            assert "audio" in call_args.kwargs
            assert call_args.kwargs["audio"] == audio_config

    def test_openai_native_passes_prediction_to_api(self) -> None:
        """Test that OpenAI native provider passes prediction to API."""
        from crewai.llms.providers.openai.completion import OpenAICompletion

        prediction_config = {"type": "content", "content": "test"}
        llm = OpenAICompletion(
            model="gpt-4o",
            prediction=prediction_config,
            api_key="test-key",
        )

        with patch.object(llm.client.chat.completions, "create") as mock_create:
            mock_create.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Test response",
                            tool_calls=None,
                        )
                    )
                ],
                usage=MagicMock(
                    prompt_tokens=10, completion_tokens=20, total_tokens=30
                ),
            )

            llm.call("Test message")

            # Verify that OpenAI API was called with prediction
            call_args = mock_create.call_args
            assert call_args is not None
            assert "prediction" in call_args.kwargs
            assert call_args.kwargs["prediction"] == prediction_config

    def test_llm_copy_preserves_responses_api_params(self) -> None:
        """Test that copying an LLM preserves Responses API parameters."""
        audio_config = {"voice": "alloy", "format": "wav"}
        prediction_config = {"type": "content", "content": "test"}

        llm = LLM(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio=audio_config,
            prediction=prediction_config,
            is_litellm=True,
        )

        llm_copy = llm.__copy__()

        assert llm_copy.modalities == ["text", "audio"]
        assert llm_copy.audio == audio_config
        assert llm_copy.prediction == prediction_config

    def test_llm_deepcopy_preserves_responses_api_params(self) -> None:
        """Test that deep copying an LLM preserves Responses API parameters."""
        import copy

        audio_config = {"voice": "alloy", "format": "wav"}
        prediction_config = {"type": "content", "content": "test"}

        llm = LLM(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio=audio_config,
            prediction=prediction_config,
            is_litellm=True,
        )

        llm_deepcopy = copy.deepcopy(llm)

        assert llm_deepcopy.modalities == ["text", "audio"]
        assert llm_deepcopy.audio == audio_config
        assert llm_deepcopy.prediction == prediction_config
        # Verify deep copy creates new objects
        assert llm_deepcopy.audio is not llm.audio
        assert llm_deepcopy.prediction is not llm.prediction

    def test_llm_responses_api_method_exists(self) -> None:
        """Test that LLM class has responses() method."""
        llm = LLM(model="gpt-4o", is_litellm=True)
        assert hasattr(llm, "responses")
        assert callable(llm.responses)

    def test_llm_aresponses_api_method_exists(self) -> None:
        """Test that LLM class has aresponses() method."""
        llm = LLM(model="gpt-4o", is_litellm=True)
        assert hasattr(llm, "aresponses")
        assert callable(llm.aresponses)

    def test_llm_responses_api_calls_litellm(self) -> None:
        """Test that responses() method calls litellm.responses."""
        llm = LLM(model="gpt-4o", is_litellm=True)

        with patch("litellm.responses") as mock_responses:
            mock_responses.return_value = {"id": "resp_123", "status": "completed"}

            result = llm.responses(input="Hello, world!")

            # Verify litellm.responses was called
            assert mock_responses.called
            call_args = mock_responses.call_args
            assert call_args is not None
            assert call_args.kwargs["model"] == "gpt-4o"
            assert call_args.kwargs["input"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_llm_aresponses_api_calls_litellm(self) -> None:
        """Test that aresponses() method calls litellm.aresponses."""
        llm = LLM(model="gpt-4o", is_litellm=True)

        with patch("litellm.aresponses") as mock_aresponses:
            mock_aresponses.return_value = {"id": "resp_123", "status": "completed"}

            result = await llm.aresponses(input="Hello, world!")

            # Verify litellm.aresponses was called
            assert mock_aresponses.called
            call_args = mock_aresponses.call_args
            assert call_args is not None
            assert call_args.kwargs["model"] == "gpt-4o"
            assert call_args.kwargs["input"] == "Hello, world!"

    def test_openai_native_responses_api_method_exists(self) -> None:
        """Test that OpenAI native provider has responses() method."""
        from crewai.llms.providers.openai.completion import OpenAICompletion

        llm = OpenAICompletion(model="gpt-4o", api_key="test-key")
        assert hasattr(llm, "responses")
        assert callable(llm.responses)

    def test_openai_native_responses_api_calls_sdk(self) -> None:
        """Test that OpenAI native provider responses() calls SDK."""
        from crewai.llms.providers.openai.completion import OpenAICompletion

        llm = OpenAICompletion(model="gpt-4o", api_key="test-key")

        with patch.object(llm.client.responses, "create") as mock_create:
            mock_create.return_value = {"id": "resp_123", "status": "completed"}

            result = llm.responses(input="Hello, world!")

            # Verify OpenAI SDK was called
            assert mock_create.called
            call_args = mock_create.call_args
            assert call_args is not None
            assert call_args.kwargs["model"] == "gpt-4o"
            assert call_args.kwargs["input"] == "Hello, world!"
