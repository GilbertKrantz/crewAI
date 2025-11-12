"""Tests for OpenAI Responses API support in CrewAI."""

from unittest.mock import patch

import pytest

from crewai.llm import LLM


class TestResponsesAPISupport:
    """Test suite for OpenAI Responses API support."""

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

    def test_llm_responses_api_with_all_parameters(self) -> None:
        """Test that responses() passes all parameters correctly."""
        llm = LLM(model="gpt-4o", is_litellm=True)

        with patch("litellm.responses") as mock_responses:
            mock_responses.return_value = {"id": "resp_123", "status": "completed"}

            result = llm.responses(
                input="Hello",
                instructions="Be helpful",
                max_output_tokens=100,
                temperature=0.7,
                previous_response_id="resp_122",
            )

            # Verify parameters were passed
            call_args = mock_responses.call_args
            assert call_args is not None
            assert call_args.kwargs["input"] == "Hello"
            assert call_args.kwargs["instructions"] == "Be helpful"
            assert call_args.kwargs["max_output_tokens"] == 100
            assert call_args.kwargs["temperature"] == 0.7
            assert call_args.kwargs["previous_response_id"] == "resp_122"

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

    def test_openai_native_responses_api_with_parameters(self) -> None:
        """Test that OpenAI native provider passes parameters correctly."""
        from crewai.llms.providers.openai.completion import OpenAICompletion

        llm = OpenAICompletion(model="gpt-4o", api_key="test-key")

        with patch.object(llm.client.responses, "create") as mock_create:
            mock_create.return_value = {"id": "resp_123", "status": "completed"}

            result = llm.responses(
                input="Test",
                instructions="System prompt",
                max_output_tokens=50,
            )

            # Verify parameters
            call_args = mock_create.call_args
            assert call_args is not None
            assert call_args.kwargs["input"] == "Test"
            assert call_args.kwargs["instructions"] == "System prompt"
            assert call_args.kwargs["max_output_tokens"] == 50
