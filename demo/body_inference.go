package main

import "fmt"

func (c config) bodyModeName() string {
	if c.bodyMode == "" {
		return "standard"
	}
	return c.bodyMode
}
func (c config) bodyInferenceArgs(args []string) []string {
	switch c.bodyModeName() {
	case "standard":
		return args
	case "no-correctives":
		return append(args, "--no-body-correctives")
	case "fast512", "fast448", "fast384":
		return append(args, "--body-intermediates=0,1,2", "--no-body-correctives", "--slim-body-intermediates", "--body-crop-size="+c.bodyModeName()[4:])
	default:
		panic("unvalidated body inference mode")
	}
}
func (c config) validateBodyMode() error {
	switch c.bodyModeName() {
	case "standard", "no-correctives", "fast512", "fast448", "fast384":
		return nil
	}
	return fmt.Errorf("body-inference must be standard, no-correctives, fast512, fast448 or fast384")
}
func (c config) inferenceLabel() string {
	if c.bodyModeName() == "standard" {
		return c.precisionLabel()
	}
	return c.precisionLabel() + " · " + c.bodyModeName() + " (approximate)"
}
