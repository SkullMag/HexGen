package server

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func assertSingleJSON(t *testing.T, body string) map[string]interface{} {
	t.Helper()
	d := json.NewDecoder(strings.NewReader(body))
	var v map[string]interface{}
	if err := d.Decode(&v); err != nil {
		t.Fatalf("invalid error JSON: %s", body)
	}
	var extra interface{}
	if err := d.Decode(&extra); err != io.EOF {
		t.Fatalf("extra JSON after response: %s", body)
	}
	return v
}

func TestForwardErrorsReturnOneResponse(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, tc := range []struct {
		name, body string
		err        error
		status     int
	}{
		{"transport", "", errors.New("stream limit exceeded"), 500},
		{"malformed", "ERROR: stream reset", nil, 500},
		{"success", `{"data":"[\"hello\",1.0]","message":"ok"}`, nil, 200},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			writeForwardedInferenceResponse(c, tc.body, tc.err)
			if w.Code != tc.status {
				t.Fatalf("status %d", w.Code)
			}
			value := assertSingleJSON(t, w.Body.String())
			if tc.status != 200 {
				if _, ok := value["error"]; !ok {
					t.Fatal("missing error")
				}
				if _, ok := value["data"]; ok {
					t.Fatal("error reported as success")
				}
			}
		})
	}
}

func TestProxyErrorHasStatusAndJSON(t *testing.T) {
	w := httptest.NewRecorder()
	ErrorHandler(w, httptest.NewRequest(http.MethodGet, "/", nil), errors.New("stream limit exceeded"))
	if w.Code != http.StatusBadGateway {
		t.Fatalf("status %d", w.Code)
	}
	value := assertSingleJSON(t, w.Body.String())
	if value["error"] != "stream limit exceeded" {
		t.Fatal(value)
	}
}
